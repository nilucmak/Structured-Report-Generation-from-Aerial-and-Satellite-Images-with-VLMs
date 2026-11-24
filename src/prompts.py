"""Loads a versioned prompt file (prompts/<version>.yaml) and fills in the text for one sample.

Input: config dict, prompt family ("native" or "json"), model family ("qwen2vl", "qwen25vl", "geovlm"), sample.
Output: the prompt text and a prompt id like "json.qwen2vl.detection.v1" (stored in every raw line).
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


def render_prompt(prompts, prompt_family, model_family, sample):
    task = sample["task"]
    classes = prompts["classes"].get(sample["dataset"], "")
    if prompt_family == "native":
        text = _fill(prompts["native"][model_family][task], {"classes": classes})
    elif prompt_family == "json":
        j = prompts["json"]
        if sample.get("time"):
            time_info = _fill(j["time_info"]["known"], {"time": str(sample["time"])})
        else:
            time_info = j["time_info"]["unknown"]
        text = _fill(j["template"], {"image_desc": j["image_desc"][task], "task_rule": j["task_rule"][task],
                                     "coord_rule": j["coord_rule"][model_family], "time_info": time_info})
        text = _fill(text, {"classes": classes})      # {classes} sits inside task_rule
    else:
        raise ValueError(f"unknown prompt family '{prompt_family}', expected one of {PROMPT_FAMILIES}")
    prompt_id = f"{prompt_family}.{model_family}.{task}.{prompts['version']}"
    return text, prompt_id
