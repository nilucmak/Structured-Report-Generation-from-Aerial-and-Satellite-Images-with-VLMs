"""Loads a versioned prompt file (prompts/<version>.yaml) and fills in the text for one sample.

Input: config dict, prompt family ("native" or "json"), model family ("qwen2vl", "qwen25vl", "geovlm"), sample.
Output: a list of prompts for the sample, each {"text", "prompt_id", "query_class"}. Usually one; a native
template containing {class} gives one prompt per vocabulary class (v2+). prompt_id is stored in every raw line,
e.g. "json.qwen2vl.detection.v2" or "native.qwen2vl.detection.v2/airplane".
"""
import yaml

from config import resolve_path

PROMPT_FAMILIES = ("native", "json")


def load_prompts(cfg, version=None):
    version = version or cfg["prompts"]["version"]
    path = resolve_path(cfg["paths"]["prompts_dir"]) / f"{version}.yaml"
    with open(path, encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    if prompts.get("version") != version:
        raise ValueError(f"{path} says version '{prompts.get('version')}', expected '{version}'")
    return prompts


def _fill(text, values):
    """Replace only the known {placeholders}; the JSON braces in the template stay untouched."""
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


def class_list(prompts, dataset):
    """Vocabulary of a dataset as a list (v1 stored it as one comma-separated string)."""
    c = prompts["classes"].get(dataset, [])
    return [x.strip() for x in c.split(",")] if isinstance(c, str) else list(c)


def render_prompts(prompts, prompt_family, model_family, sample):
    task, version = sample["task"], prompts["version"]
    classes = class_list(prompts, sample["dataset"])
    base_id = f"{prompt_family}.{model_family}.{task}.{version}"
    if prompt_family == "native":
        template = _fill(prompts["native"][model_family][task], {"classes": ", ".join(classes)})
        if "{class}" in template:   # one query per class
            return [{"text": _fill(template, {"class": c}), "prompt_id": f"{base_id}/{c}", "query_class": c}
                    for c in classes]
        return [{"text": template, "prompt_id": base_id, "query_class": None}]
    if prompt_family == "json":
        j = prompts["json"]
        if sample.get("time"):
            time_info = _fill(j["time_info"]["known"], {"time": str(sample["time"])})
        else:
            time_info = j["time_info"]["unknown"]
        text = _fill(j["template"], {"image_desc": j["image_desc"][task], "task_rule": j["task_rule"][task],
                                     "coord_rule": j["coord_rule"][model_family], "time_info": time_info})
        text = _fill(text, {"classes": ", ".join(classes)})      # {classes} sits inside task_rule
        return [{"text": text, "prompt_id": base_id, "query_class": None}]
    raise ValueError(f"unknown prompt family '{prompt_family}', expected one of {PROMPT_FAMILIES}")
