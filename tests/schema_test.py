"""Checks the report schema and build_report() on hand-written model outputs (no GPU, no data needed).

Input: nothing.
Output: prints "ALL OK" or stops at the first failed check. Also writes an example report to stdout.
Usage: python tests/schema_test.py
"""
import json
import sys
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from report_schema import ModelReport, build_report, region_of_box, report_to_json  # noqa: E402

SAMPLE = {"dataset": "dior", "sample_id": "11726", "task": "detection", "width": 800, "height": 800, "time": None}

# A model output as JSON text, the way stage 3 will receive it (boxes already in pixels)
GOOD = """{
  "summary": "Two airplanes are parked near the terminal.",
  "what": [{"class": "airplane", "count": 2, "description": "white airplanes",
            "evidence": {"box_ids": [0, 1], "regions": []}}],
  "where": [{"id": 0, "class": "airplane", "box": [256, 131, 318, 189], "region": "center"},
            {"id": 1, "class": "airplane", "box": [600, 650, 700, 760]}],
  "when": "summer 2015"
}"""


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg)


def is_invalid(text):
    try:
        ModelReport.model_validate_json(text)
        return False
    except ValidationError:
        return True


def main():
    print("Regions:")
    check(region_of_box([0, 0, 100, 100], 800, 800) == "top_left", "small box in the corner -> top_left")
    check(region_of_box([350, 350, 450, 450], 800, 800) == "center", "centred box -> center")
    check(region_of_box([700, 700, 900, 900], 800, 800) == "bottom_right", "box partly outside -> clamped")

    print("ModelReport validation:")
    mr = ModelReport.model_validate_json(GOOD)
    check(mr.what[0].class_name == "airplane", "'class' key is read into class_name")
    check(is_invalid(GOOD.replace('"center"', '"top-left-ish"')), "unknown region name rejected")
    check(is_invalid(GOOD.replace("[256, 131, 318, 189]", "[318, 131, 256, 189]")), "x1 > x2 rejected")
    check(is_invalid(GOOD.replace("[256, 131, 318, 189]", "[256, 131, 318]")), "box with 3 numbers rejected")
    check(is_invalid(GOOD.replace('"count": 2', '"count": -1')), "negative count rejected")
    check(is_invalid(GOOD.replace('"summary"', '"latitude": 41.0, "summary"')), "extra fields rejected")

    print("build_report:")
    rep = build_report(mr, SAMPLE, "qwen2vl_7b", "detection_v1")
    check(rep.when.value == "unknown" and rep.when.source == "none", "invented date replaced by 'unknown'")
    check(any(w.startswith("when_not_from_metadata") for w in rep.warnings), "invented date is reported")
    check(mr.when == "summer 2015", "model report itself is kept unchanged")
    check(rep.where[0].region == "top_center", "region recomputed from the box")
    check(any(w.startswith("region_corrected") for w in rep.warnings), "region correction is reported")
    check(rep.where[1].region == "bottom_right", "missing region filled from the box")
    check(rep.image.coordinate_system == "image_pixels_xyxy_top_left", "coordinates labelled as image pixels")

    honest = ModelReport.model_validate_json(GOOD.replace("summer 2015", "unknown"))
    check(build_report(honest, SAMPLE, "m", "v1").warnings == ["region_corrected: box 0 'center' -> 'top_center'"],
          "'unknown' from the model gives no date warning")
    with_meta = build_report(honest, {**SAMPLE, "time": "2018-05-01"}, "m", "v1")
    check(with_meta.when.value == "2018-05-01" and with_meta.when.source == "metadata", "metadata time is used")

    out = report_to_json(rep)
    check(out["what"][0]["class"] == "airplane" and "class_name" not in out["what"][0], "JSON uses 'class'")
    print("\nExample final report:")
    print(json.dumps(out, indent=2))
    print("ALL OK")


if __name__ == "__main__":
    main()
