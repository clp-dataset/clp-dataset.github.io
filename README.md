## CLP: A Real-World Dataset of Contaminated Lens Protectors for Robust Semantic Segmentation (CVPR 2026)

A benchmark for semantic segmentation and image restoration under **lens-protector
contamination**. 600 indoor/outdoor scenes are captured through a transparent
protector contaminated by different substances at four lens-to-protector distances,
yielding 4,800 degraded images paired with their clean references and dense
segmentation labels.

[![Download CLP Dataset from Google Drive](https://img.shields.io/badge/Download-Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1mB1MGo2KZKAX4Gnln7ysn6IZJpggV73A?usp=sharing)

## Overview
| | |
|---|---|
| Scenes (Case) | 600 — **train 480 / test 120** (disjoint scene split) |
| Resolution | 1000 × 750, PNG |
| Object categories | 125 (id 0 = background) |
| Instances (segmentation) | 19,203 |
| Degraded images | **4,800 = 600 scenes × 2 contamination tracks × 4 distances** |
| Clean references | 600 (one per scene) |
| License | CC BY 4.0 |

### Naming rules
Files are named `Case{ID}_{TYPE}_{DIST}.png`:
- **TYPE**: `C` = clean, `M` = mud, `W` = water, `H` = humidity, `O` = original (clean reference, no `DIST`).
- **DIST** = lens-to-protector distance in **millimeters**: `0 / 10 / 30 / 50` = **0 / 1 / 3 / 5 cm**.
- Each scene has 2 contamination tracks (clean `C` + one real contaminant `M`/`W`/`H`), each at 4 distances → 8 degraded images + 1 original.

> **Note:** segmentation masks are annotated once per scene (on the clean original) and
> shared across all degraded variants of that scene — contamination does not move objects.

---

## 1. `restoration/` — paired restoration data

Train as degraded → clean. `input/` and `target/` use **identical filenames**
(`target/X.png` is the clean ground truth for `input/X.png`).

```
restoration/
├── train/                       # 3,840 pairs
│   └── {clean,mud,water,humidity}/
│       ├── input/   Case*_*.png   (degraded)
│       └── target/  Case*_*.png   (clean GT, same filename)
└── test/                        # 960 pairs
    └── {clean,mud,water,humidity}/
        └── {0,10,30,50}/         # by distance (mm)
            ├── input/
            └── target/
```
Total = 4,800 input/target pairs.

Minimal loader:
```python
from glob import glob; import os
pairs = []
for t in ["clean","mud","water","humidity"]:
    for f in glob(f"restoration/train/{t}/input/*.png"):
        pairs.append((f, f.replace("/input/", "/target/")))
```

---

## 2. `segmentation/` — Mask2Former-ready data

Exactly the form used to train/evaluate Mask2Former (Detectron2 `sem_seg`).
Labels are grayscale PNGs where **pixel value = class id** (0–124); 255 = ignore.

```
segmentation/
├── images/{split}/   Case*.png        # 18 splits
├── labels/{split}/   Case*.png        # same splits, pixel = class id
├── _classes.csv                       # 125 classes (id, name)
├── _palette1.json                     # per-class RGB color
└── register_clp_seg.py                # Detectron2 registration (self-contained)
```

Splits (18): `train`, `test_original`, and `test_{clean,mud,humidity,water}_{0,10,30,50}`.
- `train` = 4,320 images, `test_original` = 120, `test_clean_*` = 120 each, `test_{mud,humidity,water}_*` = 40 each.

Register & train:
```python
from segmentation.register_clp_seg import register_clp_seg
register_clp_seg()                      # registers clp_semseg_<split>
# configs: DATASETS.TRAIN=("clp_semseg_train",); DATASETS.TEST=("clp_semseg_test_mud_30",)
```

---

## Citation
```bibtex
@InProceedings{Park_2026_CVPR,
    author    = {Park, Sungyong and Choi, Sooyoung and Koh, Hyunsuh and Choi, Youngjae and Kim, Heewon},
    title     = {CLP: A Real-World Dataset of Contaminated Lens Protectors for Robust Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {3794-3804}
}
```
License: **CC BY 4.0**.
