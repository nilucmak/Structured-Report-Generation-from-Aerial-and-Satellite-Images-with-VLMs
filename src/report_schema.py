"""Pydantic schema of the "What, Where, When" report and the function that builds the final report.

Input: a model's report (`ModelReport`, after the model output is parsed) + the sample from data_loaders.py.
Output: a validated final `Report`. "when" is ALWAYS taken from metadata ("unknown" if there is none),
coarse regions are computed from the boxes, and every rule violation is listed in `warnings`.

Two layers, on purpose:
- ModelReport: exactly what the model claims (kept unchanged, so stage 5 can measure e.g. invented dates).
- Report:      the final, rule-enforcing report shown to people.
Boxes are always IMAGE PIXELS (x1, y1, x2, y2, origin top-left), never geographic coordinates.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1.0"
UNKNOWN = "unknown"

# 3x3 grid over the image, plus "whole_image" for scene-level statements
Region = Literal[
    "top_left", "top_center", "top_right",
    "middle_left", "center", "middle_right",
    "bottom_left", "bottom_center", "bottom_right",
    "whole_image",
]
ChangeType = Literal["appeared", "disappeared", "modified"]


class Evidence(BaseModel):
    """What a claim is based on: ids of boxes in `where` and/or coarse regions."""
    model_config = ConfigDict(extra="forbid")
    box_ids: List[int] = Field(default_factory=list)
    regions: List[Region] = Field(default_factory=list)


class WhatItem(BaseModel):
    """One object class (or one kind of change) seen in the image."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    class_name: str = Field(alias="class")       # "class" is a Python keyword, so the attribute is class_name
    count: int = Field(ge=0)
    description: str
    change: Optional[ChangeType] = None          # only for the change-detection task
    evidence: Evidence = Field(default_factory=Evidence)


class WhereItem(BaseModel):
    """One located object. `box` is [x1, y1, x2, y2]."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    id: int
    class_name: str = Field(alias="class")
    box: List[float] = Field(min_length=4, max_length=4)
    region: Optional[Region] = None

    @field_validator("box")
    @classmethod
    def corners_in_order(cls, b):
        if b[0] > b[2] or b[1] > b[3]:
            raise ValueError("box must be [x1, y1, x2, y2] with x1 <= x2 and y1 <= y2")
        return b


class ModelReport(BaseModel):
    """What the model is asked to produce (also the schema for constrained generation in stage 4).

    Box coordinates here are already converted to image pixels by the stage-3 parsers.
    `when` is a free string so that invented dates can be detected; the prompt asks for "unknown".
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    summary: str
    what: List[WhatItem]
    where: List[WhereItem] = Field(default_factory=list)
    when: str = UNKNOWN


class When(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str                                    # a value from metadata, or "unknown"
    source: Literal["metadata", "none"]


class ImageInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: int
    height: int
    # Fixed label so nobody reads the boxes as map coordinates
    coordinate_system: Literal["image_pixels_xyxy_top_left"] = "image_pixels_xyxy_top_left"


class Report(BaseModel):
    """Final report: model claims + enforced rules + warnings."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    schema_version: str = SCHEMA_VERSION
    dataset: str
    sample_id: str
    task: Literal["detection", "change", "caption"]
    model: str
    prompt_version: str
    image: ImageInfo
    summary: str
    what: List[WhatItem]
    where: List[WhereItem]
    when: When
    warnings: List[str] = Field(default_factory=list)


def region_of_box(box, width, height):
    """Coarse 3x3 region of the box centre, e.g. a box centred in the left third, top third -> 'top_left'."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    col = min(int(3 * cx / width), 2) if cx >= 0 else 0     # 0, 1, 2 (clamped for boxes outside the image)
    row = min(int(3 * cy / height), 2) if cy >= 0 else 0
    names = [["top_left", "top_center", "top_right"],
             ["middle_left", "center", "middle_right"],
             ["bottom_left", "bottom_center", "bottom_right"]]
    return names[row][col]


def when_from_metadata(sample):
    """The only allowed source of time information: the dataset metadata (`sample['time']`)."""
    if sample.get("time"):
        return When(value=str(sample["time"]), source="metadata")
    return When(value=UNKNOWN, source="none")


def build_report(model_report, sample, model_name, prompt_version):
    """ModelReport + sample -> final Report. The model report itself is not modified."""
    warnings = []
    w, h = sample["width"], sample["height"]

    # WHEN: always from metadata. If the model wrote something else, keep a warning (measured in stage 5).
    when = when_from_metadata(sample)
    claimed = model_report.when.strip()
    if claimed.lower() != UNKNOWN and claimed != when.value:
        warnings.append(f"when_not_from_metadata: model said '{claimed}', replaced by '{when.value}'")

    # WHERE: region is recomputed from the box, so it always agrees with the pixels
    where = []
    for item in model_report.where:
        new_region = region_of_box(item.box, w, h)
        if item.region is not None and item.region != new_region:
            warnings.append(f"region_corrected: box {item.id} '{item.region}' -> '{new_region}'")
        where.append(item.model_copy(update={"region": new_region}))

    return Report(
        dataset=sample["dataset"],
        sample_id=sample["sample_id"],
        task=sample["task"],
        model=model_name,
        prompt_version=prompt_version,
        image=ImageInfo(width=w, height=h),
        summary=model_report.summary,
        what=model_report.what,
        where=where,
        when=when,
        warnings=warnings,
    )


def report_to_json(report):
    """Serialize with the public field names ("class", not "class_name")."""
    return report.model_dump(by_alias=True)
