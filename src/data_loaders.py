"""Dataset loaders: convert DIOR, VisDrone, LEVIR-CD+ and RSICD annotations into one common format.

Input: the dataset folders described in data/README.md and the settings in configs/config.yaml.
Output: a list of sample dicts per dataset, all with the same keys (see `make_sample`).

Common sample format (JSON-serializable):
    dataset      "dior" | "visdrone" | "levir_cdplus" | "rsicd"
    sample_id    unique id inside the dataset (file name without extension)
    task         "detection" | "change" | "caption"
    image_path   path relative to data_root (time 1 image for LEVIR-CD+)
    image_path_b second image (time 2) for LEVIR-CD+, otherwise None
    width, height  image size in pixels
    objects      list of {"class": str, "box": [x1, y1, x2, y2]} in PIXELS (top-left origin)
    captions     reference sentences (RSICD), otherwise []
    scene        scene category (RSICD, from file name), otherwise None
    time         acquisition time info from metadata, or None when the dataset has none
    extra        small dataset-specific info (e.g. change ratio)
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from config import dataset_root, resolve_path

# VisDrone category ids -> names (VisDrone2018-DET-toolkit README)
VISDRONE_CLASSES = {
    0: "ignored-regions", 1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor", 11: "others",
}

# DIOR's 20 classes, normalized (lowercase, no hyphens/spaces/underscores).
# Sources spell some differently ("expressway-service-area" vs "expresswayservicearea"),
# so we compare normalized names.
DIOR_CLASSES = [
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge", "chimney", "dam",
    "expresswayservicearea", "expresswaytollstation", "golffield", "groundtrackfield", "harbor",
    "overpass", "ship", "stadium", "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill",
]


def make_sample(dataset, sample_id, task, image_path, width, height, objects=None,
                captions=None, scene=None, time=None, image_path_b=None, extra=None):
    """Builds one sample in the common format (single place that defines the keys)."""
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "task": task,
        "image_path": str(image_path),
        "image_path_b": str(image_path_b) if image_path_b else None,
        "width": int(width),
        "height": int(height),
        "objects": objects or [],
        "captions": captions or [],
        "scene": scene,
        "time": time,
        "extra": extra or {},
    }


def _require(path, what):
    if not Path(path).exists():
        raise FileNotFoundError(f"{what} not found: {path}\nSee data/README.md for the expected layout.")


def _relative_to_data_root(cfg, path):
    """Store paths relative to data_root so the same JSON works locally and on Colab."""
    return Path(path).relative_to(resolve_path(cfg["paths"]["data_root"])).as_posix()


def normalize_dior_class(name):
    return name.strip().lower().replace("-", "").replace(" ", "").replace("_", "")


# ---------------------------------------------------------------- DIOR
def load_dior(cfg):
    root = dataset_root(cfg, "dior")
    split = cfg["datasets"]["dior"]["split"]
    ids_file = root / "ImageSets" / "Main" / f"{split}.txt"
    ann_dir = root / "Annotations" / "Horizontal Bounding Boxes"
    img_dir = root / ("JPEGImages-test" if split == "test" else "JPEGImages-trainval")
    for p, what in [(ids_file, "DIOR id list"), (ann_dir, "DIOR annotation folder"), (img_dir, "DIOR image folder")]:
        _require(p, what)

    ids = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
    samples = []
    for sid in ids:
        xml_path = ann_dir / f"{sid}.xml"
        img_path = img_dir / f"{sid}.jpg"
        if not xml_path.exists() or not img_path.exists():
            continue  # skip ids with a missing file (PDF risk plan: drop problematic samples)
        tree = ET.parse(xml_path).getroot()

        objects = []
        for obj in tree.findall("object"):
            raw_name = obj.findtext("name", default="")
            cls = normalize_dior_class(raw_name)
            if cls not in DIOR_CLASSES:
                raise ValueError(f"Unknown DIOR class '{raw_name}' in {xml_path}")
            bb = obj.find("bndbox")
            box = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            objects.append({"class": cls, "box": box})

        # Image size: from <size> if present, otherwise from the image header
        size = tree.find("size")
        if size is not None and size.findtext("width"):
            w, h = int(size.findtext("width")), int(size.findtext("height"))
        else:
            with Image.open(img_path) as im:
                w, h = im.size

        samples.append(make_sample("dior", sid, "detection", _relative_to_data_root(cfg, img_path),
                                   w, h, objects=objects, time=None))
    return samples


# ---------------------------------------------------------------- VisDrone
def load_visdrone(cfg):
    root = dataset_root(cfg, "visdrone")
    img_dir, ann_dir = root / "images", root / "annotations"
    _require(img_dir, "VisDrone image folder")
    _require(ann_dir, "VisDrone annotation folder")
    drop = set(cfg["datasets"]["visdrone"]["drop_categories"])

    samples = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        ann_path = ann_dir / f"{img_path.stem}.txt"
        if not ann_path.exists():
            continue
        objects = []
        for line in ann_path.read_text().splitlines():
            parts = [p for p in line.strip().split(",") if p != ""]  # some lines end with a comma
            if len(parts) < 6:
                continue
            left, top, bw, bh, score, cat = (int(float(v)) for v in parts[:6])
            if score == 0 or cat in drop:  # score 0 = ignored in official evaluation
                continue
            # VisDrone gives (left, top, width, height); convert to (x1, y1, x2, y2)
            objects.append({"class": VISDRONE_CLASSES[cat], "box": [left, top, left + bw, top + bh]})

        with Image.open(img_path) as im:  # only reads the header, fast
            w, h = im.size
        samples.append(make_sample("visdrone", img_path.stem, "detection",
                                   _relative_to_data_root(cfg, img_path), w, h, objects=objects, time=None))
    return samples


# ---------------------------------------------------------------- LEVIR-CD+
def mask_to_boxes(mask, min_area):
    """Changed pixels -> one box per connected changed region (used as 'where' evidence)."""
    labeled, _ = ndimage.label(mask > 0)
    boxes = []
    for region_id, slc in enumerate(ndimage.find_objects(labeled), start=1):
        if slc is None:
            continue
        area = int((labeled[slc] == region_id).sum())
        if area < min_area:
            continue
        ys, xs = slc
        boxes.append({"class": "change", "box": [xs.start, ys.start, xs.stop, ys.stop], "area": area})
    return boxes


def load_levir_cdplus(cfg):
    dcfg = cfg["datasets"]["levir_cdplus"]
    root = dataset_root(cfg, "levir_cdplus") / dcfg["split"]
    a_dir, b_dir, l_dir = root / "A", root / "B", root / "label"
    for p in (a_dir, b_dir, l_dir):
        _require(p, "LEVIR-CD+ folder")

    samples = []
    for a_path in sorted(a_dir.glob("*.png")):
        b_path, l_path = b_dir / a_path.name, l_dir / a_path.name
        if not b_path.exists() or not l_path.exists():
            continue
        mask = np.array(Image.open(l_path).convert("L"))  # 0 = no change, 255 = change
        h, w = mask.shape
        change_ratio = float((mask > 0).mean())
        samples.append(make_sample(
            "levir_cdplus", a_path.stem, "change",
            _relative_to_data_root(cfg, a_path), w, h,
            objects=mask_to_boxes(mask, dcfg["min_component_area"]),
            image_path_b=_relative_to_data_root(cfg, b_path),
            # The dataset has no per-pair dates; we only know A and B are two different times.
            time=None,
            extra={"label_path": _relative_to_data_root(cfg, l_path), "change_ratio": round(change_ratio, 5)},
        ))
    return samples


# ---------------------------------------------------------------- RSICD
def rsicd_scene_from_filename(filename):
    """'airport_1.jpg' -> 'airport'; names without a text prefix give 'unknown'."""
    stem = Path(filename).stem
    prefix = stem.rsplit("_", 1)[0] if "_" in stem else ""
    return prefix if prefix and not prefix.isdigit() else "unknown"


def load_rsicd(cfg):
    dcfg = cfg["datasets"]["rsicd"]
    root = dataset_root(cfg, "rsicd")
    ann_path, img_dir = root / dcfg["annotations"], root / dcfg["images_dir"]
    _require(ann_path, "RSICD annotation file")
    _require(img_dir, "RSICD image folder")

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    # The schema is not verified yet: stop with a clear message if it is not the expected one
    if not isinstance(data, dict) or "images" not in data:
        keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
        raise ValueError(f"Unexpected RSICD JSON layout. Top-level keys: {keys}. Expected 'images'.")
    first = data["images"][0] if data["images"] else {}
    missing = [k for k in ("filename", "split", "sentences") if k not in first]
    if missing:
        raise ValueError(f"RSICD entries miss keys {missing}. Keys found: {list(first.keys())}")
    if first["sentences"] and "raw" not in first["sentences"][0]:
        raise ValueError(f"RSICD sentences have no 'raw' key. Keys found: {list(first['sentences'][0].keys())}")

    samples = []
    for entry in data["images"]:
        if entry["split"] != dcfg["split"]:
            continue
        img_path = img_dir / entry["filename"]
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            w, h = im.size
        samples.append(make_sample(
            "rsicd", Path(entry["filename"]).stem, "caption", _relative_to_data_root(cfg, img_path), w, h,
            captions=[s["raw"] for s in entry["sentences"]],
            scene=rsicd_scene_from_filename(entry["filename"]), time=None,
        ))
    return samples


LOADERS = {
    "dior": load_dior,
    "visdrone": load_visdrone,
    "levir_cdplus": load_levir_cdplus,
    "rsicd": load_rsicd,
}


def load_dataset(cfg, name):
    return LOADERS[name](cfg)
