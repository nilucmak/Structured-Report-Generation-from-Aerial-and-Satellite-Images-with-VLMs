"""Smoke test: builds tiny fake versions of the 4 datasets and runs the loaders and pilot selection on them.

Input: nothing (no GPU, no downloads; fake data goes to a temporary folder that is deleted afterwards).
Output: prints "ALL OK" or stops at the first failed check.
Usage: python tests/smoke_test.py
"""
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from config import load_config  # noqa: E402
from data_loaders import DIOR_CLASSES, load_dataset  # noqa: E402

N_PER_DATASET = 15


def write_image(path, size=(64, 64)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 120, 120)).save(path)


def make_dior(root, rng):
    """30 images; spelled like mmrotate ('expressway-service-area') to test name normalization."""
    ids = [f"{i:05d}" for i in range(1, 31)]
    (root / "ImageSets" / "Main").mkdir(parents=True)
    (root / "ImageSets" / "Main" / "test.txt").write_text("\n".join(ids))
    ann = root / "Annotations" / "Horizontal Bounding Boxes"
    ann.mkdir(parents=True)
    for i, sid in enumerate(ids):
        write_image(root / "JPEGImages-test" / f"{sid}.jpg", (80, 80))
        cls = DIOR_CLASSES[i % 20].replace("expressway", "expressway-").replace("servicearea", "service-area")
        objs = "".join(
            f"<object><name>{cls}</name><bndbox><xmin>{x}</xmin><ymin>5</ymin>"
            f"<xmax>{x + 10}</xmax><ymax>20</ymax></bndbox></object>" for x in rng.sample(range(0, 60), 2))
        (ann / f"{sid}.xml").write_text(
            f"<annotation><filename>{sid}.jpg</filename><size><width>80</width><height>80</height></size>{objs}</annotation>")


def make_visdrone(root, rng):
    for i in range(25):
        name = f"img_{i:03d}"
        write_image(root / "images" / f"{name}.jpg", (100, 60))
        lines = [f"10,10,20,30,1,{1 + (i + k) % 10},0,0" for k in range(3)]
        lines += ["0,0,5,5,0,0,0,0", "1,1,5,5,1,11,0,0,"]  # ignored region + "others" (both dropped)
        (root / "annotations").mkdir(parents=True, exist_ok=True)
        (root / "annotations" / f"{name}.txt").write_text("\n".join(lines))


def make_levir(root):
    for i in range(25):
        name = f"pair_{i:03d}.png"
        for d in ("A", "B"):
            write_image(root / "test" / d / name)
        mask = np.zeros((64, 64), dtype=np.uint8)
        if i % 5:  # every 5th pair has no change -> must be filtered out
            mask[5:25, 5:25] = 255
            mask[40:50, 40:42] = 255  # 20 px region (kept, = min_component_area)
            mask[60, 60] = 255        # 1 px noise (dropped)
        (root / "test" / "label").mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(root / "test" / "label" / name)


def make_rsicd(root):
    scenes = ["airport", "beach", "farmland", "forest", "port", "school", "river", "park",
              "desert", "church", "bridge", "center", "industrial", "meadow", "pond", "stadium", "square"]
    images = []
    for i in range(60):
        fn = f"{scenes[i % len(scenes)]}_{i}.jpg"
        write_image(root / "RSICD_images" / fn, (224, 224))
        split = "test" if i < 40 else "train"
        images.append({"filename": fn, "imgid": i, "split": split,
                       "sentences": [{"raw": f"caption {k} of {fn}"} for k in range(5)]})
    for i in range(1, 4):  # real RSICD also has purely numeric names without a scene
        fn = f"{i:05d}.jpg"
        write_image(root / "RSICD_images" / fn, (224, 224))
        images.append({"filename": fn, "imgid": 100 + i, "split": "test", "sentences": [{"raw": "x"}] * 5})
    (root / "dataset_rsicd.json").write_text(json.dumps({"images": images}))


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg)


def main():
    rng = random.Random(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = tmp / "data"
        make_dior(data / "DIOR", rng)
        make_visdrone(data / "VisDrone2019-DET-val", rng)
        make_levir(data / "LEVIR-CD+")
        make_rsicd(data / "RSICD")

        # Same config as the real one, but pointing at the fake data and a temporary output file
        cfg = load_config()
        cfg["paths"]["data_root"] = str(data)
        cfg["paths"]["pilot_samples"] = str(tmp / "pilot_samples.json")
        cfg_path = tmp / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        print("Loaders:")
        dior = load_dataset(cfg, "dior")
        check(len(dior) == 30 and all(o["class"] in DIOR_CLASSES for s in dior for o in s["objects"]),
              "DIOR: 30 samples, hyphenated class names normalized")
        vis = load_dataset(cfg, "visdrone")
        check(all(len(s["objects"]) == 3 for s in vis), "VisDrone: ignored regions and 'others' dropped")
        check(vis[0]["objects"][0]["box"] == [10, 10, 30, 40], "VisDrone: (left,top,w,h) -> (x1,y1,x2,y2)")
        lev = load_dataset(cfg, "levir_cdplus")
        check(len(lev) == 25 and lev[1]["image_path_b"].endswith("B/pair_001.png"), "LEVIR-CD+: A/B pairs found")
        check(len(lev[1]["objects"]) == 2, "LEVIR-CD+: noise region dropped, 2 change boxes kept")
        check(lev[1]["objects"][0]["box"] == [5, 5, 25, 25], "LEVIR-CD+: change box in pixels")
        rs = load_dataset(cfg, "rsicd")
        check(len(rs) == 43 and len(rs[0]["captions"]) == 5, "RSICD: test split with 5 captions")
        check(rs[0]["scene"] == "airport", "RSICD: scene taken from file name")
        check(rs[-1]["scene"] == "unknown", "RSICD: numeric file name -> scene 'unknown'")
        check(all(s["time"] is None for s in dior + vis + lev + rs), "no dataset invents a time")

        print("Pilot selection:")
        cmd = [sys.executable, str(REPO / "src" / "select_pilot.py"), "--config", str(cfg_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        first = json.loads((tmp / "pilot_samples.json").read_text())
        for name, d in first["datasets"].items():
            check(len(d["samples"]) == N_PER_DATASET, f"{name}: {N_PER_DATASET} samples selected")
            ids = [s["sample_id"] for s in d["samples"]]
            check(len(set(ids)) == len(ids), f"{name}: no duplicates")
        dior_strata = [s["stratum"] for s in first["datasets"]["dior"]["samples"]]
        check(len(set(dior_strata)) == N_PER_DATASET, "DIOR: 15 different classes")
        vis_strata = {s["stratum"] for s in first["datasets"]["visdrone"]["samples"]}
        check(len(vis_strata) == 10, "VisDrone: all 10 classes represented")
        lev_ids = {s["sample_id"] for s in first["datasets"]["levir_cdplus"]["samples"]}
        check(not any(int(i[-3:]) % 5 == 0 for i in lev_ids), "LEVIR-CD+: no-change pairs excluded")
        rs_strata = [s["stratum"] for s in first["datasets"]["rsicd"]["samples"]]
        check("unknown" not in rs_strata, "RSICD: images without a scene label never selected")

        subprocess.run(cmd, check=True, capture_output=True)
        second = json.loads((tmp / "pilot_samples.json").read_text())
        check(first == second, "same seed -> identical selection")

        subprocess.run(cmd + ["--only", "rsicd"], check=True, capture_output=True)
        third = json.loads((tmp / "pilot_samples.json").read_text())
        check(third == first, "--only keeps the other datasets in the file")
    print("ALL OK")


if __name__ == "__main__":
    main()
