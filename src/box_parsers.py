"""Turns raw model text into boxes in ORIGINAL image pixels [x1, y1, x2, y2] (top-left origin) or a ModelReport.

Input: one raw record from results/raw/*.jsonl (needs family, task, prompt_family, image size, input size, raw_output).
Output: parse_record() -> {"ok", "boxes", "report", "error", "text"}. Pure Python, no GPU (tested in tests/parser_test.py).

Coordinate conventions (verified, see CLAUDE.md):
- Qwen2-VL:   <|object_ref_start|>label<|object_ref_end|><|box_start|>(x1,y1),(x2,y2)<|box_end|>, values in [0, 1000).
- Qwen2.5-VL: JSON [{"bbox_2d": [x1, y1, x2, y2], "label": ...}], absolute pixels of the RESIZED model input
              (input size = image_grid_thw * 14), rescaled to the original image like the official cookbook.
- GeoVLM-R1:  [[x_left, y_top, x_right, y_bottom, theta]] on the 448x448 input, theta in degrees
              (format and rotation formula from GeoVLM-R1-Toolkit GeoVLM_Eval/rs_grounding_description/eval.py).
"""
import json
import math
import re

from pydantic import ValidationError

from report_schema import ModelReport

SPECIAL_TOKEN = re.compile(r"<\|[a-z_]+\|>")
NUM = r"-?\d+(?:\.\d+)?"
QWEN2VL_BOX = re.compile(
    r"(?:<\|object_ref_start\|>(.*?)<\|object_ref_end\|>)?\s*"        # optional label before the box
    rf"<\|box_start\|>\s*\(\s*({NUM})\s*,\s*({NUM})\s*\)\s*,\s*\(\s*({NUM})\s*,\s*({NUM})\s*\)\s*<\|box_end\|>",
    re.S)
PLAIN_PAIR_BOX = re.compile(rf"\(\s*({NUM})\s*,\s*({NUM})\s*\)\s*,\s*\(\s*({NUM})\s*,\s*({NUM})\s*\)")
FLAT_JSON_OBJECT = re.compile(r"\{[^{}]*\}")                         # a JSON object without nested braces
LEADING_FILLER = {"there", "are", "is", "in", "on", "at", "of", "the", "image", "see", "i", "can", "be", "seen", "by", "nearby",
                  "additionally", "also", "right", "left", "close", "to"}
GEOVLM_BOX = re.compile(rf"\[\s*({NUM}(?:\s*,\s*{NUM}){{3,4}})\s*\]")  # [a, b, c, d] or [a, b, c, d, theta]


def strip_special(text):
    return SPECIAL_TOKEN.sub("", text)


def answer_part(text):
    """GeoVLM-R1 may answer as <think>...</think><answer>...</answer>; keep only the answer if present."""
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.S)     # also works if the output was truncated
    return m.group(1) if m else text


def _r(v):
    return round(float(v), 1)


def scale_box(box, sx, sy):
    """Multiply x by sx and y by sy, and put the corners in order (x1 <= x2, y1 <= y2)."""
    x1, y1, x2, y2 = (float(b) for b in box)
    xs, ys = sorted([x1 * sx, x2 * sx]), sorted([y1 * sy, y2 * sy])
    return [_r(xs[0]), _r(ys[0]), _r(xs[1]), _r(ys[1])]


def obb_to_hbb(x_left, y_top, x_right, y_bottom, theta):
    """Oriented box -> enclosing horizontal box, same rotation as the official GeoVLM-R1 evaluation:
    corners of [x_left, y_top, x_right, y_bottom] are rotated by theta degrees around the box centre
    (rotated = (corner - centre) @ [[cos, -sin], [sin, cos]] + centre), then min/max is taken."""
    cx, cy = (x_left + x_right) / 2, (y_top + y_bottom) / 2
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    xs, ys = [], []
    for x, y in [(x_left, y_top), (x_right, y_top), (x_right, y_bottom), (x_left, y_bottom)]:
        dx, dy = x - cx, y - cy
        xs.append(dx * c + dy * s + cx)          # row vector times the matrix, as in eval.py
        ys.append(-dx * s + dy * c + cy)
    return [min(xs), min(ys), max(xs), max(ys)]


def model_input_size(record):
    """(height, width) of the image the model saw (the last image for change pairs)."""
    h, w = record["input_sizes"][-1]
    return h, w


# ---------------------------------------------------------------- native formats

def parse_qwen2vl_native(text, width, height):
    boxes, label = [], None
    for m in QWEN2VL_BOX.finditer(text):
        if m.group(1) is not None:
            label = m.group(1).strip()         # one label can be followed by several boxes
        x1, y1, x2, y2 = (float(v) for v in m.groups()[1:])
        boxes.append({"label": label, "box": scale_box([x1, y1, x2, y2], width / 1000, height / 1000)})
    if not boxes:                              # fallback: coordinates written without the box tokens
        for m in PLAIN_PAIR_BOX.finditer(strip_special(text)):
            boxes.append({"label": None, "box": scale_box([float(v) for v in m.groups()], width / 1000, height / 1000)})
    return boxes


def parse_qwen25vl_native(text, width, height, input_h, input_w):
    boxes = []
    for m in FLAT_JSON_OBJECT.finditer(strip_special(text)):  # scanning objects one by one survives truncated lists
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        b = obj.get("bbox_2d")
        if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
            boxes.append({"label": obj.get("label"), "box": scale_box(b, width / input_w, height / input_h)})
    return boxes


def _label_before(segment):
    """Heuristic label for a GeoVLM box: the last few words of the text right before it
    (e.g. '... there are two gray small dry-cargo-ships [[...' -> 'two gray small dry-cargo-ships')."""
    segment = re.split(r"[.;:]|\band\b|\bwith\b|\bnear\b", segment)[-1]
    words = re.sub(r"[\[\],]", " ", segment).split()[-4:]
    while words and words[0].lower() in LEADING_FILLER:   # 'are two white airplanes' -> 'two white airplanes'
        words = words[1:]
    return " ".join(words) or None


def parse_geovlm_native(text, width, height, size=448):
    answer = strip_special(answer_part(text))
    boxes, label, prev_end = [], None, 0
    for m in GEOVLM_BOX.finditer(answer):
        new_label = _label_before(answer[prev_end:m.start()])
        label = new_label or label             # boxes in the same [[...], [...]] group share the label
        prev_end = m.end()
        nums = [float(v) for v in m.group(1).split(",")]
        theta = nums[4] if len(nums) == 5 else 0.0
        hbb = obb_to_hbb(*nums[:4], theta)
        boxes.append({"label": label, "box": scale_box(hbb, width / size, height / size),
                      "obb_448": nums if len(nums) == 5 else nums + [0.0]})
    return boxes


# ---------------------------------------------------------------- our JSON report

def extract_json_object(text):
    """First '{' to last '}' of the answer, after removing special tokens and ``` fences."""
    t = strip_special(answer_part(text))
    t = re.sub(r"```(?:json)?", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(t[start:end + 1])


def convert_report_box(box, record):
    """Box in the model's own convention -> original image pixels. Non-numeric boxes are returned
    unchanged so that schema validation rejects them (and the error is counted in stage 5)."""
    if not (isinstance(box, list) and len(box) in (4, 5) and all(isinstance(v, (int, float)) for v in box)):
        return box
    w, h, fam = record["image_width"], record["image_height"], record["family"]
    if fam == "geovlm":
        size = record.get("geovlm_size", 448)
        hbb = obb_to_hbb(*box[:4], box[4]) if len(box) == 5 else box
        return scale_box(hbb, w / size, h / size)
    if len(box) != 4:
        return box
    if fam == "qwen2vl":
        return scale_box(box, w / 1000, h / 1000)
    if fam == "qwen25vl":
        in_h, in_w = model_input_size(record)
        return scale_box(box, w / in_w, h / in_h)
    raise ValueError(f"unknown model family '{fam}'")


def parse_report_json(record):
    """-> (ModelReport or None, error message or None)."""
    try:
        obj = extract_json_object(record["raw_output"])
    except (ValueError, json.JSONDecodeError) as e:
        return None, f"json_error: {e}"
    if isinstance(obj, dict) and isinstance(obj.get("where"), list):
        for item in obj["where"]:
            if isinstance(item, dict) and "box" in item:
                item["box"] = convert_report_box(item["box"], record)
    try:
        return ModelReport.model_validate(obj), None
    except ValidationError as e:
        return None, f"schema_error: {e.error_count()} error(s): " + "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()[:5])


# ---------------------------------------------------------------- dispatch

def parse_record(record):
    """One raw line -> parsed result. 'boxes' are always original image pixels."""
    raw, fam = record["raw_output"], record["family"]
    text = strip_special(answer_part(raw)).strip()
    if record["prompt_family"] == "json":
        report, err = parse_report_json(record)
        return {"ok": report is not None, "boxes": [], "error": err, "text": text,
                "report": report.model_dump(by_alias=True) if report else None}
    w, h = record["image_width"], record["image_height"]
    if record["task"] == "caption":
        boxes = []
    elif fam == "qwen2vl":
        boxes = parse_qwen2vl_native(raw, w, h)
    elif fam == "qwen25vl":
        in_h, in_w = model_input_size(record)
        boxes = parse_qwen25vl_native(raw, w, h, in_h, in_w)
    elif fam == "geovlm":
        boxes = parse_geovlm_native(raw, w, h, record.get("geovlm_size", 448))
    else:
        raise ValueError(f"unknown model family '{fam}'")
    return {"ok": True, "boxes": boxes, "report": None, "error": None, "text": text}
