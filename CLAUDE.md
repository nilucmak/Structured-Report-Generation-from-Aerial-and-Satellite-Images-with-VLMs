# CLAUDE.md — Project context (read at the start of every session)

Official scope: `docs/2209A_arastirma_onerisi.pdf` (TÜBİTAK 2209-A research proposal, 2026 term 1).
The PDF is git-ignored.

## Working rules (set by the user)
- **Language:** talk to the user in **Turkish**. Everything in the repository is **English**
  (README, code, comments, file headers, commit messages, this file).
- Do the stages **in order**. At the end of each stage STOP, explain in plain Turkish what was done,
  which files were created and why. Do not continue until the user says "devam".
- Every file starts with a 2–3 line header: what it does, its input, its output. Add short comments
  to non-obvious lines.
- **Never guess** model APIs, special tokens or coordinate formats. Verify from Hugging Face model
  cards / official docs / dataset sources. If unsure, tell the user.
- **Never download datasets.** Document where/how to download and the expected folder layout
  (`data/README.md`); the user downloads.
- Keep it simple: no class hierarchies, no frameworks, no unnecessary abstraction. Readable Python.
- All settings (model names, paths, seed, max_new_tokens) live in one file: `configs/config.yaml`.
- For any major design decision, present options first and let the user choose.
- Commit only when the user asks.
- Commit messages have **no** `Co-Authored-By` or other AI attribution lines (user's choice).
- No personal names (student, advisor) in repository files.

## Project summary (from the PDF)
- Topic: generate structured JSON reports with **"What, Where, When"** (ne / nerede / ne_zaman)
  fields from aerial and satellite images using open-weight VLMs, and measure their accuracy.
- **No training / fine-tuning.** Pretrained weights only; zero-shot and instruction-based inference.
- Tasks and datasets:
  - Object recognition + localization: **DIOR** (satellite), **VisDrone** (drone)
  - Temporal change analysis: **LEVIR-CD+** (image pairs)
  - Scene description: **RSICD**
- Full study (PDF): 400 samples (DIOR 100, VisDrone 100, LEVIR-CD+ 100 pairs, RSICD 100).
  **Current goal: 60-sample pilot** (15 per dataset, fixed prompts, 3 models; report n, mean ± std).
- Sampling (PDF): samples must represent different object classes (DIOR, VisDrone), clear
  structural/environmental changes (LEVIR-CD+) and different scene categories (RSICD).
- Metrics (PDF): Precision, Recall, F1, IoU (mAP where appropriate); hallucination/omission for
  descriptions; report factual accuracy, completeness, spatial accuracy, evidence consistency;
  JSON schema validity (target ≥ 95% valid); human evaluation; inference time and VRAM; mean ± std.
- Hard rules (PDF sections 2 and 5):
  - Pixel coordinates are **never** presented as geographic coordinates.
  - Time/location not present in metadata is **never** generated (`ne_zaman` = "bilinmiyor" if absent).
  - Unverifiable information is never presented as a definite finding. The system supports, not replaces, human judgement.
- Hardware: Google Colab, NVIDIA T4 (~15 GB VRAM). Code must also run locally without a GPU (tests).
  The EVREN platform is in the PDF but out of scope for now.
- Feasibility results (PDF Table 1): Qwen2-VL-7B 4-bit ~6 GB VRAM; on DIOR #10 it drew one wide box
  around 6 airplanes. GeoVLM-R1 grounding found the 6 airplanes separately and produced OBBs (~6.5 GB).
  GeoVLM-R1 CD_Caption ~7.5 GB, short text.

## Models and verified facts (as of 2026-11-23)
| Model | HF id | Class | Note |
|---|---|---|---|
| Qwen2-VL-7B-Instruct | `Qwen/Qwen2-VL-7B-Instruct` | `Qwen2VLForConditionalGeneration` | 4-bit NF4 (bitsandbytes). License apache-2.0. |
| GeoVLM-R1 grounding | `ibm-research/GeoVLM-R1`, subfolder `GeoVLM_R1_grounding` | `Qwen2_5_VLForConditionalGeneration` | Images resized to 448×448. |
| GeoVLM-R1 CD_Caption | `ibm-research/GeoVLM-R1`, subfolder `GeoVLM_R1_CD_Caption` | `Qwen2_5_VLForConditionalGeneration` | Change captioning. |
| GeoVLM-R1 Image_caption | `ibm-research/GeoVLM-R1`, subfolder `GeoVLM_R1_Image_caption` | `Qwen2_5_VLForConditionalGeneration` | Used for RSICD (stage 3 choice). |
| Qwen2.5-VL-3B-Instruct | `Qwen/Qwen2.5-VL-3B-Instruct` | `Qwen2_5_VLForConditionalGeneration` | Base model of GeoVLM-R1 (verified). License: Qwen Research License. |

- **GeoVLM-R1 base = Qwen2.5-VL-3B-Instruct** — VERIFIED: model card `base_model: Qwen/Qwen2.5-VL-3B-Instruct`;
  paper §4.1 "We select Qwen2.5VL-3B-Instruct ... as our base model"; config `model_type: qwen2_5_vl`.
  Training: Qwen2.5-VL-3B → SFT on EarthDial-Instruct → GRPO.
- The GeoVLM-R1 HF repo has no root config; each task is a separate 3B checkpoint (~7.5 GB) in a subfolder.
- GeoVLM-R1 license is **conflicting**: HF card "without a license", GitHub toolkit CC BY-NC 4.0,
  paper CC BY-NC-SA 4.0 → assume non-commercial research use only.
- GeoVLM-R1 grounding output — VERIFIED (toolkit `GeoVLM_Eval/rs_grounding_description/eval.py` + GRPO training
  data `GRPO_GeoChat_grounding.jsonl`): free text with box groups `[[x_left, y_top, x_right, y_bottom, theta], ...]`,
  integers on the 448×448 input, theta in degrees. OBB corners = rotate the 4 corners around the box centre with
  `(corner-centre) @ [[cos,-sin],[sin,cos]]` (implemented in `src/box_parsers.py:obb_to_hbb`). Even the training
  data has broken values (e.g. 4395) → out-of-image boxes are expected and counted in stage 5.
- GeoVLM-R1 prompts: GRPO training appends a `<think>/<answer>` template; the official inference script sends the
  plain question → our parser accepts both. Native prompts in `prompts/v1.yaml` are verbatim training prompts.
- GeoVLM-R1 official inference: `qwen_vl_utils.process_vision_info`, `resized_height/width = 448`,
  `max_new_tokens=128`. Card uses bfloat16, toolkit script float16. Toolkit: https://github.com/IBM/GeoVLM-R1-Toolkit
  Its `generation_config.json` sets `use_cache: false`, `repetition_penalty: 1.05` → we pass `use_cache=True` and an
  explicit `repetition_penalty` from config (1.0 for all models).
- Qwen2-VL grounding — VERIFIED: tokenizer has `<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`,
  `<|box_end|>` (ids 151646–151649), all marked **special** → decode with `skip_special_tokens=False` or the
  tokens disappear. Coordinates normalized to [0, 1000), `(x1,y1),(x2,y2)` (arXiv 2409.12191 §2.2.1).
- Qwen2.5-VL grounding — VERIFIED (official cookbook `spatial_understanding.ipynb`, commit f61c3cd, 2025-01):
  JSON `{"bbox_2d": [x1, y1, x2, y2], "label": ...}` in **absolute pixels of the resized model input**
  (input size = `image_grid_thw * 14`); rescale by original/input size.
- T4 has no native bfloat16 → float16 (watch for overflow/NaN). No flash-attention-2 → `sdpa`.
- `transformers==4.49.0` pin (GeoVLM-R1 configs saved with it). Not yet run on Colab.
- Possible train/test contamination (tell the advisor): GeoVLM-R1 was trained on RSICD **train** (no overlap with
  our 15 test samples, VERIFIED); CD_Caption on LEVIR-CC (derived from LEVIR-CD, may overlap LEVIR-CD+ test,
  NOT verified); grounding on GeoChat data (may contain DIOR images, NOT verified).

## Dataset facts (stage 1; see `data/README.md`)
- DIOR (VERIFIED on the real files): Pascal VOC XML with only filename/source/size/segmented/object(name, pose, bndbox).
  **No date/time fields.** Images 800×800. Test split: 11,738 images; all 20 classes present; 192,518 boxes in all XMLs.
  Raw spelling is mixed case, e.g. `Expressway-Service-area`, `Expressway-toll-station` → loader normalizes
  (lowercase, no hyphens). Layout `DIOR/{JPEGImages-test, Annotations/Horizontal Bounding Boxes, ImageSets/Main/test.txt}`.
  License: CC BY-NC 4.0 (author page; sources conflict).
- VisDrone2019-DET: one txt per image, `<left>,<top>,<w>,<h>,<score>,<category>,<trunc>,<occl>`;
  score 0 = ignored; categories 0 ignored-regions, 1 pedestrian … 10 motor, 11 others. Pilot uses the val split (548 images; layout VERIFIED). Images are crowded: the 15 pilot images have 21–161 objects.
- LEVIR-CD+ (layout VERIFIED on the real zip; files are named `train_<n>.png` even in test/): `LEVIR-CD+/{train,test}/{A,B,label}/*.png`, same basename across A/B/label; mask 0 = no change,
  255 = change; 1024×1024; 985 pairs (637 train / 348 test). **No per-pair dates** (only a global span,
  sources conflict: 2002–2018 vs 2002–2020) → `ne_zaman` stays "bilinmiyor"; A/B only gives order.
  Whether A is always the earlier image is assumed from the dataset convention, not verified.
- RSICD (VERIFIED on the real files): 10,921 images, 224×224; `dataset_rsicd.json` = `{"images": [{"filename", "imgid",
  "split", "sentences": [{"tokens", "raw", "imgid", "sentid"}]}]}` (Karpathy style). Splits train 8734 / val 1094 / test 1093.
  30 scene prefixes; 921 files have numeric names (`00001.jpg`) → scene "unknown", excluded from the pilot.
  `raw` captions are lowercase with a space before the period. **10,197 of 10,921 images have duplicated
  captions** (fewer than 5 unique) → decide in stage 5 whether to deduplicate references for BLEU/CIDEr.
- None of the four datasets provides per-image acquisition dates (as far as verified).

## Report schema (stage 2, decided with the user)
- JSON keys are **English**: `what` (= ne), `where` (= nerede), `when` (= ne_zaman), `evidence` (= kanit);
  unknown time is `"unknown"`; regions are a 3×3 grid (`top_left` … `bottom_right`) + `whole_image`.
- Two layers: `ModelReport` = what the model claims (kept unchanged for measurement; also the stage-4
  constrained-generation schema) → `build_report()` → final `Report`.
- **The model fills every field, the code enforces the rules** (user's choice): final `when` is always
  from metadata (`source: metadata|none`), regions are recomputed from boxes, and each correction is
  written to `warnings` (e.g. `when_not_from_metadata`) so stage 5 can count them.
- Boxes are image pixels `[x1,y1,x2,y2]`, top-left origin; `image.coordinate_system` is the fixed label
  `image_pixels_xyxy_top_left`. Extra keys (e.g. `latitude`) are rejected by the schema.
- Count-vs-boxes and boxes-inside-image are NOT rejected by the schema on purpose: stage 5 measures them.

## Inference design (stage 3, decided with the user)
- Two prompt families per sample: `native` (each model's documented format) and `json` (ModelReport, boxes in the
  model's own coordinate convention, converted by `src/box_parsers.py`). Stage 4 adds constrained `json`.
- max_new_tokens per task: detection 1024, change 512, caption 256; raw line has `truncated`. No sample dropped.
- Qwen images: min_pixels 256·28², max_pixels 1280·28² (model-card example). GeoVLM: 448×448.
- GeoVLM-R1 checkpoint per task: detection → grounding, change → CD_Caption, caption → Image_caption.
- Detection prompts give the dataset's class vocabulary (never the image's ground-truth classes).
- **Prompt v1 → v2 (first Colab tiny run, 2026-11-24):**
  - With v1, both Qwen models drew one box over the whole DIOR image and labelled the scene (`airport`).
  - Qwen2-VL copied the special-token template in the prompt literally (`class name`, and it echoed the change prompt).
  - v2 (user's choice): Qwen native detection asks **one class per query**, which gives 480 prompts per Qwen model
    and also measures hallucinated classes. There is no template text in the prompts. GeoVLM prompts and the json family are unchanged.
  - v1 files are kept and reported as v1 (prompt sensitivity).
  - No official Qwen2-VL grounding prompt exists in its README or model card; the v2 wording is our design.
- **Qwen2-VL vision-encoder bug (found by a Colab probe, 2026-11-24):** with default bitsandbytes settings the vision
  encoder was quantized to 4-bit too and the model was blind: it called a street an "abstract painting", saw 0 cars and
  drew one box over the whole image. Fix: `quant_skip_modules: [visual, lm_head]` in config (vision layers become
  `Linear`, language layers `Linear4bit`; 7.2 GB after loading). **All v1/v2 Qwen2-VL results are invalid.**
- **Prompt v3:**
  - v2 Qwen detection answered "none"/[] for all 30 queries (even 63 cars). The "if there is none, answer none" sentence
    made the models skip.
  - v3 uses wordings that the probe confirmed produce real boxes. Qwen2.5-VL: the official cookbook sentence. Qwen2-VL:
    "Detect the bounding box of every {class} in this image." There is no escape sentence.
  - The json template uses `<...>` descriptions; v2 outputs had copied the example values `name`/`short text`.
- Real outputs confirmed the formats: GeoVLM `[[x1,y1,x2,y2,θ]]` on 448 with no `<think>` tags; Qwen2.5-VL `bbox_2d`
  in model-input pixels (800 px is resized to 812); Qwen2-VL emits its box tokens. T4 peak VRAM: Qwen2-VL 7.3 GB,
  Qwen2.5-VL 8.6 GB, GeoVLM 7.9 GB.
- Raw outputs keep special tokens; parsing is separate. Colab gets only `pilot_data.zip` (~65 MB, `src/pack_pilot_data.py`).

## Deviations from the PDF (tell the advisor)
1. **Third model:** the PDF says "at least one of TEOChat, SkyEyeGPT, RSGPT". The pilot uses
   Qwen2.5-VL-3B-Instruct (GeoVLM-R1's base model → fair measure of the RS fine-tuning effect).
   The PDF's "other technically suitable models" may allow this; confirm with the advisor.
2. Pilot is 60 samples; the full study is 400.
3. EVREN experiments postponed.
4. Qwen2-VL-7B "4-bit": only the language model is 4-bit; the vision encoder stays float16 (it was blind in 4-bit on T4).

## Stage status
- [x] Stage 0 — Setup (CLAUDE.md, folder layout, requirements, README, git init)
- [x] Stage 1 — Dataset loaders + 15-per-dataset pilot selection (`results/pilot_samples.json`, all 4 datasets selected)
- [x] Stage 2 — Pydantic report schema (`src/report_schema.py`, tests in `tests/schema_test.py`)
- [~] Stage 3 — code done (`src/run_inference.py`, `src/box_parsers.py`, `prompts/v1.yaml`, `v2.yaml`, `v3.yaml`, `notebooks/stage3_inference.ipynb`, `src/show_results.py`,
  tests in `tests/parser_test.py`); tiny run done on Colab with v1 (transformers 4.49.0 works); v3 tiny run + full pilot pending
- [ ] Stage 4 — Free vs Outlines-constrained generation
- [ ] Stage 5 — Metrics and one script that rebuilds all tables
- [ ] Stage 6 — Human evaluation template + Cohen's kappa

## Folder layout
```
configs/config.yaml   single configuration file
data/                 datasets (git-ignored; user downloads, see data/README.md)
docs/                 2209-A proposal (PDF git-ignored)
prompts/              versioned prompt templates
src/                  Python scripts (plain functions)
tests/                local smoke tests on tiny fake data (no GPU, no downloads)
notebooks/            Colab notebook (clones the repo, calls the scripts)
results/              pilot_samples.json, raw/*.jsonl, tables
```
