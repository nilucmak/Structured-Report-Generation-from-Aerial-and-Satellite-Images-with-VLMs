# Structured Report Generation from Aerial and Satellite Images with VLMs

Experiment code for a bachelor's thesis / TÜBİTAK 2209-A project. Outputs of open-weight
vision-language models (VLMs) are turned into JSON reports with "What, Where, When" fields, and
their accuracy is measured. There is no training; only inference with pretrained weights.

## Models
- `Qwen/Qwen2-VL-7B-Instruct` (4-bit NF4)
- `Qwen/Qwen2.5-VL-3B-Instruct` (base model of GeoVLM-R1)
- `ibm-research/GeoVLM-R1` → subfolders `GeoVLM_R1_grounding` and `GeoVLM_R1_CD_Caption`

## Datasets and tasks
| Dataset | Task |
|---|---|
| DIOR | Object recognition + localization (satellite) |
| VisDrone2019-DET | Object recognition + localization (drone) |
| LEVIR-CD+ | Temporal change analysis (image pairs) |
| RSICD | Scene description |

Datasets are not included. Download instructions and the expected layout: [data/README.md](data/README.md).

## Folder layout
```
configs/config.yaml   single configuration file (model names, paths, seed, max_new_tokens)
data/                 datasets (git-ignored)
docs/                 project proposal (PDF git-ignored)
prompts/              versioned prompt templates
src/                  Python scripts
tests/                smoke tests on tiny fake data (no GPU, no downloads)
notebooks/            Colab notebook
results/              pilot sample list, raw outputs (raw/*.jsonl), tables
```

## Setup
Local (no GPU, for testing):
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
On Colab (T4 GPU), the notebook in `notebooks/` will clone this repository and call the scripts.

## Usage
```bash
python tests/smoke_test.py          # checks loaders and sampling on fake data
python tests/schema_test.py         # checks the report schema and prints an example report
python tests/parser_test.py         # checks box parsers, OBB->HBB and prompt rendering
python src/select_pilot.py          # picks 15 samples per dataset -> results/pilot_samples.json
python src/pack_pilot_data.py       # copies only the 60 pilot samples -> pilot_data.zip (~65 MB) for Colab
python src/run_inference.py --model qwen2vl_7b --family native --dry-run   # prompts + image check, no GPU
```
Inference runs on Colab: `notebooks/stage3_inference.ipynb` (upload `pilot_data.zip` to Google Drive first).
The notebook has three parts (setup, run, results); the results part (`src/show_results.py`) draws ground-truth
and model boxes on the images and prints a summary table per model.
Each run appends to `results/raw/<model>__<native|json>__<prompt version>.jsonl`; finished samples are skipped.

Prompt families: `native` uses each model's own documented output format (fair test of localization);
`json` asks for the report schema of `src/report_schema.py`. Prompts are versioned in `prompts/<version>.yaml`.

## Notes
- Pixel coordinates are never presented as geographic coordinates.
- The `when` field of the final report is filled only from metadata; otherwise it is "unknown".
  If a model invents a date, it is replaced and recorded in `warnings`.
- Raw outputs are stored unchanged (special tokens kept); `src/box_parsers.py` converts every box to
  original image pixels `[x1, y1, x2, y2]`. GeoVLM-R1 oriented boxes are converted to horizontal boxes
  with the rotation formula of the official GeoVLM-R1 evaluation code.
- GeoVLM-R1 weights are for non-commercial research use only; Qwen2.5-VL-3B uses the Qwen Research License.
- Code license: MIT (see `LICENSE`). Model weights and datasets keep their own licenses (see above and `data/README.md`).
