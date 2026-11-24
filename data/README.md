# Datasets: where to download and the expected layout

Datasets are **not** committed (see `.gitignore`). Download them yourself and place them under
`data/` (or set `paths.data_root` in `configs/config.yaml`, e.g. to a Google Drive folder on Colab).
For the pilot only the splits below are needed.

All layouts below were checked on the real downloads (2026-11-24). The loaders fail with a
clear message if a folder or key is missing.

---

## 1. DIOR — object detection (satellite)
- Official page: https://gcheng-nwpu.github.io/ (section "DIOR and DIOR-R datasets")
  - Google Drive: https://drive.google.com/open?id=1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC
  - Baidu: https://pan.baidu.com/s/1iLKT0JQoKXEJTGNxt5lSMg
- Needed for the pilot: `JPEGImages-test`, `Annotations` (horizontal boxes), `ImageSets/Main/test.txt`.
- License: CC BY-NC 4.0 according to the author page (other mirrors state different licenses).
- Downloading the whole Google Drive folder gives **several** zips (`DIOR-<date>-1-001.zip`,
  `-002`, ...). All parts are needed; one part may only contain e.g. `DIOR/Annotations.zip`.
  The inner zips (`Annotations.zip`, `JPEGImages-test.zip`, ...) must be extracted too.

Expected layout (folder names from MMRotate's DIOR guide; note the spaces):
```
data/DIOR/
├── JPEGImages-test/                  <id>.jpg
├── Annotations/
│   └── Horizontal Bounding Boxes/    <id>.xml   (Pascal VOC XML)
└── ImageSets/Main/test.txt           one image id per line
```
Class names in the XML are mixed case (e.g. `Expressway-Service-area`); the loader normalizes them.
The XML has no date/time fields. The Drive download also contains `ImageSets.zip` and `Annotations.zip`
inside `DIOR/`: extract them in place. `JPEGImages-test/` and `JPEGImages-trainval/` go inside `DIOR/`.

## 2. VisDrone2019-DET — object detection (drone)
- Official repo: https://github.com/VisDrone/VisDrone-Dataset ("Task 1: Object Detection in Images")
  - **valset (0.07 GB)** — Google Drive: https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing
  - Baidu: https://pan.baidu.com/s/1jdK_dAxRJeF2Xi50IoML1g
- Annotation format (from the DET toolkit README):
  `<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>`
  — `score` 0 means "ignored in evaluation"; category 0 = ignored regions, 11 = others.
- License: not stated on the official repo.

Expected layout:
```
data/VisDrone2019-DET-val/
├── images/        <name>.jpg
└── annotations/   <name>.txt   (same name as the image)
```

## 3. LEVIR-CD+ — change detection (image pairs)
- Distributed with S2Looking: https://github.com/S2Looking/Dataset ("LEVIR-CD+ can also be downloaded at")
  - Google Drive: https://drive.google.com/drive/folders/1lJ5GteQy5Z2LVayrKYZOn7_QPImlYP8C?usp=sharing
  - Baidu: https://pan.baidu.com/s/1b0WWlYAzTMPUCAfdxyyWTg (password: obg6)
- 985 pairs, 1024×1024 px, 0.5 m/pixel; train 637 / test 348.
- Mask: single-channel PNG, 0 = no change, 255 = change.
- File names are `train_<n>.png` in **both** splits (e.g. `test/A/train_833.png`); this is how the dataset ships.
- **No per-pair acquisition dates** are provided, only a global span. So `ne_zaman` stays
  "bilinmiyor" (unknown); we only know A and B are two different times.
- License: academic use only, no commercial use (LEVIR-CD page); Google Earth terms apply.

Expected layout (from torchgeo's LEVIR-CD+ loader). The archive may contain a nested
`LEVIR-CD+/LEVIR-CD+/` folder and a `__MACOSX` folder: move the inner folder up and delete `__MACOSX`.
```
data/LEVIR-CD+/
└── test/
    ├── A/       <name>.png   (time 1)
    ├── B/       <name>.png   (time 2, same file name)
    └── label/   <name>.png   (change mask)
```

## 4. RSICD — scene captioning
- Official repo: https://github.com/201528014227051/RSICD_optimal
  (the README recommends the cloud links because of Git LFS limits)
  - Google Drive: https://drive.google.com/open?id=0B1jt7lJDEXy3aE90cG9YSl9ScUk
  - Baidu: http://pan.baidu.com/s/1bp71tE3
- Needed: `RSICD_images.zip` (extract into `data/RSICD/`) **and** `dataset_rsicd.json` (captions; a separate file in the same repo / Drive folder).
- 10,921 images, 224×224, 5 captions each; file names like `airport_1.jpg` (scene name prefix, 30 scenes).
  921 files have purely numeric names (`00001.jpg` ...) with no scene label; the pilot skips them.
- License: not stated.

Expected layout:
```
data/RSICD/
├── RSICD_images/        airport_1.jpg, ...
└── dataset_rsicd.json
```
JSON layout (checked on the real file): `{"images": [{"filename", "imgid", "split", "sentences": [{"tokens", "raw", "imgid", "sentid"}]}]}`.
Both `RSICD_images/` and `dataset_rsicd.json` go inside `data/RSICD/`.

---

After downloading, check that everything is found:
```bash
python src/select_pilot.py --check
```

For Colab, only the pilot files are needed. This copies them (images, annotations, masks, captions;
~65 MB) into `pilot_data/` and `pilot_data.zip` with the same layout; upload the zip to Google Drive:
```bash
python src/pack_pilot_data.py
```
