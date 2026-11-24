"""Runs one model on the pilot samples with one prompt family and appends raw outputs to a JSONL file.

Input: configs/config.yaml, results/pilot_samples.json, prompts/<version>.yaml, images under paths.data_root.
Output: results/raw/<model>__<prompt family>__<prompt version>.jsonl, one line per prompt (per sample, or
per sample and class for per-class prompts): raw text, seconds,
peak VRAM (GB), number of new tokens and a truncated flag. Already finished samples are skipped (safe to re-run
after a Colab disconnect). Parsing happens later (src/box_parsers.py), so the raw text is never changed.

Usage:  python src/run_inference.py --model qwen2vl_7b --family native [--datasets dior rsicd] [--limit 2]
        python src/run_inference.py --model geovlm_r1 --family json --dry-run    # no GPU: prints prompts only
"""
import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from config import load_config, resolve_path
from data_loaders import load_dataset
from prompts import PROMPT_FAMILIES, load_prompts, render_prompts

DATASETS = ("dior", "visdrone", "levir_cdplus", "rsicd")


# ---------------------------------------------------------------- samples and messages (no torch needed)

def pilot_samples(cfg, datasets):
    """Full sample dicts (from data_loaders.py) for the pilot ids, in the order of pilot_samples.json."""
    pilot = json.loads(resolve_path(cfg["paths"]["pilot_samples"]).read_text())
    out = []
    for name in datasets:
        ids = [s["sample_id"] for s in pilot["datasets"][name]["samples"]]
        by_id = {s["sample_id"]: s for s in load_dataset(cfg, name)}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise FileNotFoundError(f"{name}: pilot samples not found in the data folder: {missing[:5]}")
        out += [by_id[i] for i in ids]
    return out


def image_files(cfg, sample):
    """Absolute image paths: one image, or [earlier, later] for change pairs."""
    root = resolve_path(cfg["paths"]["data_root"])
    paths = [sample["image_path"]] + ([sample["image_path_b"]] if sample.get("image_path_b") else [])
    return [str(root / p) for p in paths]


def build_messages(cfg, family, files, prompt):
    """Chat message in the qwen_vl_utils format. Resize keys are read by process_vision_info."""
    im = cfg["images"]
    if family == "geovlm":   # official GeoVLM-R1 inference: fixed 448x448
        imgs = [{"type": "image", "image": f, "resized_height": im["geovlm_size"], "resized_width": im["geovlm_size"]}
                for f in files]
    else:                    # Qwen models: keep the aspect ratio, limit the number of pixels
        imgs = [{"type": "image", "image": f, "min_pixels": im["qwen_min_pixels"], "max_pixels": im["qwen_max_pixels"]}
                for f in files]
    return [{"role": "user", "content": imgs + [{"type": "text", "text": prompt}]}]


def output_path(cfg, model_key, prompt_family, version):
    return resolve_path(cfg["paths"]["raw_dir"]) / f"{model_key}__{prompt_family}__{version}.jsonl"


def done_keys(path):
    """(dataset, sample_id, prompt_id) keys already in the output file."""
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            keys.add((r["dataset"], r["sample_id"], r["prompt_id"]))
    return keys


# ---------------------------------------------------------------- model loading and generation (GPU)

def checkpoint_of(mcfg, task):
    """GeoVLM-R1 has one checkpoint (subfolder) per task; Qwen models have one checkpoint for everything."""
    return mcfg.get("subfolder_per_task", {}).get(task)


def load_model(cfg, model_key, task):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, \
        Qwen2VLForConditionalGeneration

    mcfg = cfg["models"][model_key]
    path, sub = mcfg["hf_id"], checkpoint_of(mcfg, task)
    if sub:  # download only this task's subfolder, then load it like a normal local model
        from huggingface_hub import snapshot_download
        path = str(Path(snapshot_download(mcfg["hf_id"], allow_patterns=[f"{sub}/*"])) / sub)
    dtype = getattr(torch, cfg["generation"]["dtype"])
    kwargs = {"torch_dtype": dtype, "device_map": "auto", "attn_implementation": "sdpa"}  # no flash-attn on T4
    if mcfg["quantization"] == "nf4_4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    cls = Qwen2VLForConditionalGeneration if mcfg["family"] == "qwen2vl" else Qwen2_5_VLForConditionalGeneration
    model = cls.from_pretrained(path, **kwargs).eval()
    processor = AutoProcessor.from_pretrained(path)
    return model, processor, sub


def generate(cfg, model, processor, messages, max_new_tokens):
    """Greedy generation for one message. Special tokens are KEPT in the decoded text for every model
    (Qwen2-VL's <|box_start|> etc. are special tokens and would otherwise be deleted)."""
    import torch
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
    # Size of each image as the model saw it: grid (t, h, w) in 14-pixel patches
    input_sizes = [[int(h) * 14, int(w) * 14] for _, h, w in inputs["image_grid_thw"].tolist()]

    gpu = torch.cuda.is_available()
    if gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=cfg["generation"]["do_sample"],
                             repetition_penalty=cfg["generation"]["repetition_penalty"],
                             temperature=None, top_p=None, top_k=None,   # unset sampling defaults (greedy)
                             use_cache=True)   # GeoVLM-R1's generation_config.json sets use_cache=false (very slow)
    if gpu:
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    eos = set(model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, list)
              else [model.generation_config.eos_token_id])
    n_new = int(len(new_ids))
    return {
        "raw_output": processor.decode(new_ids, skip_special_tokens=False),
        "n_new_tokens": n_new,
        "truncated": n_new >= max_new_tokens and int(new_ids[-1]) not in eos,  # limit hit before end of answer
        "seconds": round(seconds, 3),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if gpu else None,
        "input_sizes": input_sizes,
    }


# One inference function per model (the differences are the image size rule and the prompt wording,
# both handled through the family; kept separate so each model's settings are easy to read and change).
def infer_qwen2vl(cfg, model, processor, sample, prompt, max_new_tokens):
    return generate(cfg, model, processor, build_messages(cfg, "qwen2vl", image_files(cfg, sample), prompt),
                    max_new_tokens)


def infer_qwen25vl(cfg, model, processor, sample, prompt, max_new_tokens):
    return generate(cfg, model, processor, build_messages(cfg, "qwen25vl", image_files(cfg, sample), prompt),
                    max_new_tokens)


def infer_geovlm(cfg, model, processor, sample, prompt, max_new_tokens):
    return generate(cfg, model, processor, build_messages(cfg, "geovlm", image_files(cfg, sample), prompt),
                    max_new_tokens)


INFER = {"qwen2vl": infer_qwen2vl, "qwen25vl": infer_qwen25vl, "geovlm": infer_geovlm}


def environment():
    import torch
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


# ---------------------------------------------------------------- main loop

def jobs_for(prompts, prompt_family, family, samples):
    """(sample, prompt item) pairs: usually one per sample, one per class for per-class prompts."""
    return [(s, item) for s in samples for item in render_prompts(prompts, prompt_family, family, s)]


def run(cfg, model_key, prompt_family, datasets, limit=None, dry_run=False):
    mcfg = cfg["models"][model_key]
    family = mcfg["family"]
    prompts = load_prompts(cfg)
    samples = pilot_samples(cfg, datasets)
    if limit:
        samples = [s for name in datasets for s in [x for x in samples if x["dataset"] == name][:limit]]
    out_path = output_path(cfg, model_key, prompt_family, prompts["version"])
    done = done_keys(out_path)
    jobs = jobs_for(prompts, prompt_family, family, samples)
    todo = [(s, it) for s, it in jobs if (s["dataset"], s["sample_id"], it["prompt_id"]) not in done]

    if dry_run:
        shown = set()
        for s in samples:
            missing = [f for f in image_files(cfg, s) if not Path(f).exists()]
            if missing:
                raise FileNotFoundError(f"image not found: {missing}")
        for s, it in jobs:
            if s["task"] not in shown:   # print the first prompt of each task
                shown.add(s["task"])
                n = sum(1 for x, _ in jobs if x is s)
                print(f"--- {it['prompt_id']} ({s['dataset']} {s['sample_id']}, {n} prompt(s) for this sample, "
                      f"checkpoint: {checkpoint_of(mcfg, s['task'])})")
                print(it["text"])
        print(f"\nDry run OK: {len(samples)} samples, {len(jobs)} prompts, all images found, "
              f"{len(todo)} still to run -> {out_path}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = environment()
    # GeoVLM-R1 needs a different checkpoint per task, so jobs are grouped by checkpoint
    groups = {}
    for s, it in todo:
        groups.setdefault(checkpoint_of(mcfg, s["task"]), []).append((s, it))
    if not groups:
        print(f"[{model_key}] nothing to do")
    for sub, group in groups.items():
        model, processor, _ = load_model(cfg, model_key, group[0][0]["task"])
        import torch
        model_vram = round(torch.cuda.memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None
        torch.manual_seed(cfg["seed"])
        for i, (s, it) in enumerate(group, 1):
            max_new = cfg["generation"]["max_new_tokens"][s["task"]]
            res = INFER[family](cfg, model, processor, s, it["text"], max_new)
            record = {
                "dataset": s["dataset"], "sample_id": s["sample_id"], "task": s["task"],
                "model": model_key, "family": family, "hf_id": mcfg["hf_id"], "checkpoint": sub,
                "quantization": mcfg["quantization"], "dtype": cfg["generation"]["dtype"],
                "prompt_family": prompt_family, "prompt_version": prompts["version"], "prompt_id": it["prompt_id"],
                "query_class": it["query_class"], "prompt": it["text"], "max_new_tokens": max_new,
                "do_sample": cfg["generation"]["do_sample"], "repetition_penalty": cfg["generation"]["repetition_penalty"],
                "image_width": s["width"], "image_height": s["height"], "geovlm_size": cfg["images"]["geovlm_size"],
                **res, "model_vram_gb": model_vram, "env": env,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            with open(out_path, "a", encoding="utf-8") as f:   # append line by line: nothing lost on disconnect
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            q = f" [{it['query_class']}]" if it["query_class"] else ""
            print(f"[{model_key}/{sub}] {i}/{len(group)} {s['dataset']} {s['sample_id']}{q}: "
                  f"{res['seconds']} s, {res['n_new_tokens']} tokens, peak {res['peak_vram_gb']} GB"
                  + (" TRUNCATED" if res["truncated"] else ""))
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Done -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", required=True, help="key under models: in config.yaml")
    ap.add_argument("--family", required=True, choices=PROMPT_FAMILIES)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--limit", type=int, default=None, help="only the first N pilot samples per dataset")
    ap.add_argument("--dry-run", action="store_true", help="check images and print prompts; no model is loaded")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    if args.model not in cfg["models"]:
        raise SystemExit(f"unknown model '{args.model}', choose from {list(cfg['models'])}")
    run(cfg, args.model, args.family, args.datasets, args.limit, args.dry_run)


if __name__ == "__main__":
    main()
