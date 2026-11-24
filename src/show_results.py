"""Quick visual check of the pilot results: draws ground-truth (green) and model (red) boxes on the images,
shows captions next to the reference captions, prints a summary table per model and one example report.

Input: config (paths.raw_dir, prompts.version), results/raw/*__<version>.jsonl, the pilot samples.
Output: matplotlib figures and pandas tables (shown in the Colab notebook). This is a QUICK LOOK
(class-agnostic box matching at IoU >= 0.5), not the final metrics of stage 5.
"""
import json
import textwrap
from pathlib import Path

import pandas as pd
from PIL import Image

from box_parsers import parse_record
from config import resolve_path
from report_schema import ModelReport, build_report, report_to_json
from run_inference import DATASETS, image_files, pilot_samples

IOU_THRESHOLD = 0.5


def load_records(cfg, model_key, prompt_family):
    """All raw lines of one model and prompt family for the current prompt version."""
    path = resolve_path(cfg["paths"]["raw_dir"]) / f"{model_key}__{prompt_family}__{cfg['prompts']['version']}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def per_sample(records):
    """Merge the lines of one sample (per-class prompts give several lines) -> {(dataset, id): result}."""
    out = {}
    for r in records:
        p = parse_record(r)
        s = out.setdefault((r["dataset"], r["sample_id"]), {"boxes": [], "texts": [], "seconds": 0.0, "vram": 0.0,
                                                            "truncated": False, "ok": True, "report": None})
        s["boxes"] += p["boxes"]
        if p["text"] and not p["boxes"] and p["text"].lower() not in ("none", "[]"):
            s["texts"].append(p["text"])
        s["seconds"] += r["seconds"]
        s["vram"] = max(s["vram"], r["peak_vram_gb"] or 0.0)
        s["truncated"] |= r["truncated"]
        s["ok"] &= p["ok"]
        s["report"] = p["report"] or s["report"]
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def count_matches(gt_boxes, pred_boxes, thr=IOU_THRESHOLD):
    """Greedy one-to-one matching, highest IoU first; each GT box can be matched once."""
    pairs = sorted(((iou(g, p), i, j) for i, g in enumerate(gt_boxes) for j, p in enumerate(pred_boxes)), reverse=True)
    used_g, used_p, n = set(), set(), 0
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in used_g and j not in used_p:
            used_g.add(i), used_p.add(j)
            n += 1
    return n


def models_of(cfg):
    return list(cfg["models"])


def summary(cfg):
    """One row per model and dataset: boxes found, quick precision/recall, time, VRAM, valid JSON rate."""
    samples = pilot_samples(cfg, DATASETS)
    rows = []
    for m in models_of(cfg):
        native, js = per_sample(load_records(cfg, m, "native")), per_sample(load_records(cfg, m, "json"))
        for ds in DATASETS:
            ss = [s for s in samples if s["dataset"] == ds]
            done = [s for s in ss if (ds, s["sample_id"]) in native]
            if not done:
                continue
            gt = sum(len(s["objects"]) for s in done)
            pred = sum(len(native[(ds, s["sample_id"])]["boxes"]) for s in done)
            hit = sum(count_matches([o["box"] for o in s["objects"]],
                                    [b["box"] for b in native[(ds, s["sample_id"])]["boxes"]]) for s in done)
            res = [native[(ds, s["sample_id"])] for s in done]
            js_done = [js[(ds, s["sample_id"])] for s in ss if (ds, s["sample_id"]) in js]
            rows.append({
                "model": m, "dataset": ds, "samples": f"{len(done)}/{len(ss)}",
                "GT boxes": gt, "model boxes": pred, f"matched (IoU>={IOU_THRESHOLD})": hit,
                "precision": round(hit / pred, 2) if pred else None,
                "recall": round(hit / gt, 2) if gt else None,
                "sec/sample": round(sum(r["seconds"] for r in res) / len(res), 1),
                "max VRAM GB": round(max(r["vram"] for r in res), 2),
                "truncated": sum(r["truncated"] for r in res),
                "valid JSON": f"{sum(r['ok'] for r in js_done)}/{len(js_done)}" if js_done else "-",
            })
    return pd.DataFrame(rows)


def _draw(ax, img_path, gt_boxes, pred_boxes, title):
    from matplotlib.patches import Rectangle
    ax.imshow(Image.open(img_path).convert("RGB"))
    for b in gt_boxes:
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color="lime", lw=1.5))
    for b in pred_boxes:
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color="red", lw=1.2))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _wrap(text, width=45, max_lines=6):
    lines = textwrap.wrap(text or "-", width)
    return "\n".join(lines[:max_lines]) + (" ..." if len(lines) > max_lines else "")


def show(cfg, dataset, n=3):
    """First n pilot samples of a dataset: ground truth and every model's native output side by side."""
    import matplotlib.pyplot as plt
    models = models_of(cfg)
    results = {m: per_sample(load_records(cfg, m, "native")) for m in models}
    for s in [x for x in pilot_samples(cfg, [dataset])][:n]:
        files, key = image_files(cfg, s), (dataset, s["sample_id"])
        gt = [o["box"] for o in s["objects"]]
        cols = 1 + len(models) + (1 if s["task"] == "change" else 0)
        fig, axes = plt.subplots(1, cols, figsize=(4.2 * cols, 5.2))
        i = 0
        if s["task"] == "change":   # earlier image first, then the later image with the boxes
            _draw(axes[0], files[0], [], [], "A (earlier)")
            i = 1
        if s["task"] == "caption":
            ref = s["captions"][0] if s["captions"] else "-"
            _draw(axes[i], files[-1], [], [], f"reference:\n{_wrap(ref)}")
        else:
            _draw(axes[i], files[-1], gt, [], f"ground truth: {len(gt)} boxes")
        for k, m in enumerate(models, start=i + 1):
            r = results[m].get(key)
            if r is None:
                axes[k].axis("off")
                axes[k].set_title(f"{m}\n(not run yet)", fontsize=9)
                continue
            boxes = [b["box"] for b in r["boxes"]]
            hit = count_matches(gt, boxes)
            title = f"{m}\n{len(boxes)} boxes, {hit} matched" if s["task"] != "caption" else m
            if r["texts"]:
                title += "\n" + _wrap(" ".join(r["texts"]))
            _draw(axes[k], files[-1], gt if s["task"] != "caption" else [], boxes, title)
        fig.suptitle(f"{dataset} / {s['sample_id']}   (green = ground truth, red = model)", fontsize=11)
        plt.tight_layout()
        plt.show()


def example_report(cfg, model_key, dataset):
    """Final 'what / where / when' report of the first pilot sample whose json output was valid."""
    js = per_sample(load_records(cfg, model_key, "json"))
    for s in pilot_samples(cfg, [dataset]):
        r = js.get((dataset, s["sample_id"]))
        if r and r["ok"]:
            rep = build_report(ModelReport.model_validate(r["report"]), s, model_key, cfg["prompts"]["version"])
            return report_to_json(rep)
    return f"no valid JSON report from {model_key} on {dataset} yet"
