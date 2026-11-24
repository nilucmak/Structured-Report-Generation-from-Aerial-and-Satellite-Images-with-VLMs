"""Copies ONLY the files of the 60 pilot samples into a small folder + zip, for upload to Google Drive / Colab.

Input: configs/config.yaml, results/pilot_samples.json, the full datasets under paths.data_root.
Output: pilot_data/ (same layout as data/, see data/README.md) and pilot_data.zip (both git-ignored).
The loaders skip files that are missing, so on Colab they only see the pilot samples.
Usage: python src/pack_pilot_data.py [--out pilot_data]
"""
import argparse
import shutil
from pathlib import Path

from config import load_config, resolve_path
from run_inference import DATASETS, pilot_samples


def pilot_files(cfg, sample):
    """Data files of one sample, relative to data_root (images, annotation, change mask)."""
    files = [sample["image_path"]]
    if sample.get("image_path_b"):
        files.append(sample["image_path_b"])
    if sample["extra"].get("label_path"):
        files.append(sample["extra"]["label_path"])
    ds = cfg["datasets"]
    if sample["dataset"] == "dior":
        files.append(f"{ds['dior']['root']}/Annotations/Horizontal Bounding Boxes/{sample['sample_id']}.xml")
    if sample["dataset"] == "visdrone":
        files.append(f"{ds['visdrone']['root']}/annotations/{sample['sample_id']}.txt")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pilot_data")
    args = ap.parse_args()
    cfg = load_config()
    src_root = resolve_path(cfg["paths"]["data_root"])
    out = resolve_path(args.out)
    if out.exists():
        shutil.rmtree(out)

    samples = pilot_samples(cfg, DATASETS)
    rel_files = {f for s in samples for f in pilot_files(cfg, s)}
    ds = cfg["datasets"]
    # Files every loader needs even for a subset: the DIOR id list and the RSICD caption file
    rel_files.add(f"{ds['dior']['root']}/ImageSets/Main/{ds['dior']['split']}.txt")
    rel_files.add(f"{ds['rsicd']['root']}/{ds['rsicd']['annotations']}")
    for rel in sorted(rel_files):
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_root / rel, dst)

    zip_path = shutil.make_archive(str(out), "zip", root_dir=out)
    size_mb = Path(zip_path).stat().st_size / 1e6
    print(f"{len(samples)} samples, {len(rel_files)} files -> {out} and {zip_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
