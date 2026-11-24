"""Checks the box parsers, OBB->HBB conversion and prompt rendering on hand-written model outputs (no GPU).

Input: nothing (prompt rendering reads prompts/v1.yaml and configs/config.yaml).
Output: prints "ALL OK" or stops at the first failed check.
Usage: python tests/parser_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from box_parsers import obb_to_hbb, parse_record  # noqa: E402
from config import load_config  # noqa: E402
from prompts import load_prompts, render_prompt  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg)


def close(a, b, tol=0.6):
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def rec(family, raw, task="detection", prompt_family="native", w=800, h=800, input_sizes=((812, 812),)):
    return {"family": family, "raw_output": raw, "task": task, "prompt_family": prompt_family,
            "image_width": w, "image_height": h, "input_sizes": [list(s) for s in input_sizes], "geovlm_size": 448}


def main():
    print("OBB -> HBB:")
    check(close(obb_to_hbb(10, 20, 30, 40, 0), [10, 20, 30, 40]), "theta 0 keeps the box")
    check(close(obb_to_hbb(0, 0, 40, 20, 90), [10, -10, 30, 30]), "theta 90 swaps width and height around the centre")
    x1, y1, x2, y2 = obb_to_hbb(0, 0, 20, 20, 45)
    check(close([x2 - x1, y2 - y1], [28.28, 28.28], 0.01), "theta 45 square -> diagonal-sized box")

    print("Qwen2-VL native (special tokens, 0-1000):")
    raw = ("<|object_ref_start|>airplane<|object_ref_end|><|box_start|>(100,200),(150,260)<|box_end|>"
           "<|box_start|>(500,500),(600,560)<|box_end|>"
           "<|object_ref_start|>ship<|object_ref_end|><|box_start|>(0,0),(1000,1000)<|box_end|><|im_end|>")
    p = parse_record(rec("qwen2vl", raw, w=800, h=600))
    check(len(p["boxes"]) == 3, "3 boxes found")
    check(p["boxes"][0]["box"] == [80.0, 120.0, 120.0, 156.0], "0-1000 scaled to 800x600 pixels")
    check(p["boxes"][1]["label"] == "airplane", "box without its own label inherits the previous label")
    check(p["boxes"][2]["box"] == [0.0, 0.0, 800.0, 600.0], "1000 -> full image size")
    p = parse_record(rec("qwen2vl", "the airplane is at (100,200),(150,260)", w=1000, h=1000))
    check(p["boxes"][0]["box"] == [100.0, 200.0, 150.0, 260.0], "fallback: coordinates without box tokens")

    print("Qwen2.5-VL native (bbox_2d, model-input pixels):")
    raw = '```json\n[\n {"bbox_2d": [0, 0, 406, 406], "label": "vehicle"},\n {"bbox_2d": [406, 406, 812, 812], "label": "ship"},\n {"bbox_2d": [1, 2'
    p = parse_record(rec("qwen25vl", raw, input_sizes=[(812, 812)]))
    check(len(p["boxes"]) == 2, "truncated last object is skipped, the complete ones are kept")
    check(p["boxes"][1]["box"] == [400.0, 400.0, 800.0, 800.0], "812 input px -> 800 original px")
    p = parse_record(rec("qwen25vl", '[{"bbox_2d": [10, 10, 20, 20], "label": "car"}]', w=2000, h=1500,
                         input_sizes=[(784, 1036)]))
    check(close(p["boxes"][0]["box"], [19.3, 19.1, 38.6, 38.3], 0.1), "non-square input rescaled per axis")

    print("GeoVLM-R1 native (OBB on 448x448):")
    raw = ("<think>I see planes.</think><answer>In the image there are two white airplanes "
           "[[44, 44, 88, 88, 0], [100, 100, 120, 140, 90]] near a building [[200, 200, 224, 224, 0]] .</answer>")
    p = parse_record(rec("geovlm", raw, w=800, h=800))
    check(len(p["boxes"]) == 3, "3 boxes found inside <answer>")
    check(p["boxes"][0]["box"] == [78.6, 78.6, 157.1, 157.1], "448 grid scaled to 800 px")
    check(p["boxes"][0]["label"] == "two white airplanes", "label = words right before the box group")
    check(p["boxes"][1]["label"] == "two white airplanes", "second box of the same group keeps the label")
    check(p["boxes"][2]["label"] == "a building", "next group gets its own label")
    check(p["boxes"][1]["obb_448"] == [100, 100, 120, 140, 90], "original OBB is kept")
    check(close(p["boxes"][1]["box"], [x * 800 / 448 for x in [90, 110, 130, 130]], 0.1), "rotated box -> HBB")
    p = parse_record(rec("geovlm", "Nothing has changed.<|im_end|>", task="change"))
    check(p["boxes"] == [] and p["text"] == "Nothing has changed.", "text answer without boxes")

    print("JSON report family:")
    good = ('```json\n{"summary": "Two cars.", "what": [{"class": "car", "count": 2, "description": "red cars",'
            ' "evidence": {"box_ids": [0, 1], "regions": []}}], "where": [{"id": 0, "class": "car", "box": [0, 0, 500, 500]},'
            ' {"id": 1, "class": "car", "box": [500, 500, 1000, 1000], "region": "center"}], "when": "unknown"}\n```')
    p = parse_record(rec("qwen2vl", good, prompt_family="json", w=800, h=600))
    check(p["ok"] and p["report"]["where"][1]["box"] == [400.0, 300.0, 800.0, 600.0], "qwen2vl JSON box 0-1000 -> px")
    p = parse_record(rec("geovlm", good.replace("[0, 0, 500, 500]", "[0, 0, 224, 224, 0]"), prompt_family="json"))
    check(p["ok"] and p["report"]["where"][0]["box"] == [0.0, 0.0, 400.0, 400.0], "geovlm 5-number box -> HBB px")
    p = parse_record(rec("qwen25vl", "Sorry, I cannot do that.", prompt_family="json"))
    check(not p["ok"] and p["error"].startswith("json_error"), "no JSON -> json_error")
    p = parse_record(rec("qwen25vl", good.replace('"count": 2', '"count": "two"'), prompt_family="json"))
    check(not p["ok"] and p["error"].startswith("schema_error"), "wrong type -> schema_error")
    p = parse_record(rec("qwen25vl", good.replace("[0, 0, 500, 500]", '"top left"'), prompt_family="json"))
    check(not p["ok"] and "where.0.box" in p["error"], "non-numeric box is rejected by the schema")

    print("Prompt rendering:")
    cfg = load_config()
    prompts = load_prompts(cfg)
    for task, ds in [("detection", "dior"), ("detection", "visdrone"), ("change", "levir_cdplus"), ("caption", "rsicd")]:
        sample = {"dataset": ds, "task": task, "time": None}
        for fam in ("qwen2vl", "qwen25vl", "geovlm"):
            for pf in ("native", "json"):
                text, pid = render_prompt(prompts, pf, fam, sample)
                assert "{classes}" not in text and "{task_rule}" not in text and "{coord_rule}" not in text, pid
                assert pid == f"{pf}.{fam}.{task}.v1", pid
    check(True, "all 24 prompt combinations render with no placeholder left")
    text, _ = render_prompt(prompts, "json", "qwen2vl", {"dataset": "dior", "task": "detection", "time": None})
    check('"when" must be "unknown"' in text and "storage tank" in text, "JSON prompt: unknown time + class list")
    text, _ = render_prompt(prompts, "json", "qwen2vl", {"dataset": "dior", "task": "detection", "time": "2018-05-01"})
    check('"2018-05-01"' in text, "metadata time is passed to the prompt when it exists")
    print("ALL OK")


if __name__ == "__main__":
    main()
