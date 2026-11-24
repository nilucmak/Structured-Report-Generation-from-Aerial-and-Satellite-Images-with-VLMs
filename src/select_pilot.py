"""Selects the pilot samples (15 per dataset by default) with a fixed seed and saves their ids.

Input: datasets under data/ (see data/README.md) and configs/config.yaml (seed, sample count).
Output: results/pilot_samples.json with the chosen sample ids and the stratum each was chosen for.
Usage: python src/select_pilot.py [--check] [--only dior visdrone ...] [--config path]
"""
import argparse
import json
import random
from collections import Counter

from config import DEFAULT_CONFIG, load_config, resolve_path
from data_loaders import LOADERS, load_dataset


def strata_of(sample):
    """Which groups a sample belongs to, used to spread the pilot over classes / scenes."""
    if sample["task"] == "detection":
        return sorted({o["class"] for o in sample["objects"]})  # every class present in the image
    if sample["task"] == "caption":
        return [sample["scene"]]
    return ["change"]  # LEVIR-CD+: one group


def stratified_pick(samples, n, rng):
    """Round-robin over strata in a random (seeded) order, one random unused sample per stratum.

    With more strata than n (DIOR: 20 classes, RSICD: 30 scenes) every pick has a different stratum.
    With fewer (VisDrone: 10 classes) every stratum is used once, then the round starts again.
    """
    by_stratum = {}
    for s in sorted(samples, key=lambda x: x["sample_id"]):  # sort first: result must not depend on file order
        for st in strata_of(s):
            by_stratum.setdefault(st, []).append(s)

    order = sorted(by_stratum)
    rng.shuffle(order)
    chosen, used = [], set()
    while len(chosen) < n:
        added = False
        for st in order:
            if len(chosen) >= n:
                break
            pool = [s for s in by_stratum[st] if s["sample_id"] not in used]
            if not pool:
                continue
            pick = rng.choice(pool)
            used.add(pick["sample_id"])
            chosen.append({"sample_id": pick["sample_id"], "stratum": st, "n_objects": len(pick["objects"])})
            added = True
        if not added:  # not enough samples left
            break
    return chosen


def candidates_for(cfg, name, samples):
    """Pilot filters: detection images need at least one object; LEVIR-CD+ needs a clear change."""
    if name in ("dior", "visdrone"):
        return [s for s in samples if s["objects"]]
    if name == "levir_cdplus":
        min_ratio = cfg["datasets"]["levir_cdplus"]["min_change_ratio"]
        return [s for s in samples if s["extra"]["change_ratio"] >= min_ratio]
    if name == "rsicd":
        # 921 RSICD files have purely numeric names (e.g. 00001.jpg) and so no scene label
        return [s for s in samples if s["scene"] != "unknown"]
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--only", nargs="+", choices=list(LOADERS), help="run only these datasets")
    parser.add_argument("--check", action="store_true", help="only load the datasets and print statistics")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n = cfg["pilot_samples_per_dataset"]
    out = {"seed": cfg["seed"], "samples_per_dataset": n, "datasets": {}}

    for name in args.only or list(LOADERS):
        samples = load_dataset(cfg, name)
        cands = candidates_for(cfg, name, samples)
        strata = Counter(st for s in cands for st in strata_of(s))
        print(f"[{name}] loaded {len(samples)} samples, {len(cands)} candidates, {len(strata)} strata")
        if args.check:
            print("   strata (images per group):", dict(strata.most_common()))
            continue

        rng = random.Random(f"{cfg['seed']}:{name}")  # separate, reproducible stream per dataset
        picked = stratified_pick(cands, n, rng)
        if len(picked) < n:
            print(f"   WARNING: only {len(picked)} samples could be selected")
        print("   picked strata:", dict(Counter(p["stratum"] for p in picked)))
        out["datasets"][name] = {
            "split": cfg["datasets"][name].get("split"),
            "n_candidates": len(cands),
            "samples": picked,
        }

    if not args.check:
        path = resolve_path(cfg["paths"]["pilot_samples"])
        if args.only and path.exists():
            # Keep the other datasets already in the file when only some are re-run
            old = json.loads(path.read_text(encoding="utf-8"))
            out["datasets"] = {**old.get("datasets", {}), **out["datasets"]}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
