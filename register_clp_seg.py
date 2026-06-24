# register_clp_seg.py
# CLP segmentation dataset -> Detectron2 / Mask2Former registration.
# Self-contained: paths are resolved relative to this file's directory.
#
# Usage:
#   from register_clp_seg import register_clp_seg
#   register_clp_seg()                      # uses this folder as root
#   # then in configs: DATASETS.TRAIN = ("clp_semseg_train",)
#   #                  DATASETS.TEST  = ("clp_semseg_test_mud_30", ...)
import os, glob, csv, json
from functools import partial
import numpy as np, cv2
from detectron2.data import DatasetCatalog, MetadataCatalog

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH     = os.path.join(_HERE, "_classes.csv")
PALETTE_JSON = os.path.join(_HERE, "_palette1.json")
IGNORE       = {"NA"}
IGNORE_VAL   = 255

with open(PALETTE_JSON) as f:
    palette_dict = json.load(f)

id2name, stuff_classes = {}, []
with open(CSV_PATH, newline="") as f:
    next(f)  # header
    for cid, cname in csv.reader(f):
        cid = int(cid); cname = cname.strip()
        id2name[cid] = cname
        if cname not in IGNORE:
            stuff_classes.append(cname)

train_id_lut = np.full(256, IGNORE_VAL, np.uint8)
_next = 0
for cid, cname in id2name.items():
    if cname not in IGNORE:
        train_id_lut[cid] = _next; _next += 1

stuff_colors = [palette_dict[c] for c in stuff_classes]

def _get_pairs(img_dir, lbl_dir, lut=train_id_lut):
    files = glob.glob(os.path.join(img_dir, "**", "*.*"), recursive=True)
    recs = []
    for f in sorted(files):
        if f.endswith(".remap.png"):
            continue
        base  = os.path.basename(f).rsplit(".", 1)[0]
        lbl_f = os.path.join(lbl_dir, base + ".png")
        assert os.path.exists(lbl_f), f"missing {lbl_f}"
        lbl = cv2.imread(lbl_f, cv2.IMREAD_GRAYSCALE)
        tmp = lbl_f + ".remap.png"
        cv2.imwrite(tmp, lut[lbl])  # on-the-fly train-id remap (ignore->255)
        recs.append({"file_name": f, "sem_seg_file_name": tmp, "image_id": base})
    return recs

SPLITS = [
    "train", "test_original",
    "test_clean_0", "test_clean_10", "test_clean_30", "test_clean_50",
    "test_mud_0", "test_mud_10", "test_mud_30", "test_mud_50",
    "test_humidity_0", "test_humidity_10", "test_humidity_30", "test_humidity_50",
    "test_water_0", "test_water_10", "test_water_30", "test_water_50",
]

def register_clp_seg(root=_HERE):
    for sp in SPLITS:
        name  = f"clp_semseg_{sp}"
        img_d = os.path.join(root, "images", sp)
        lbl_d = os.path.join(root, "labels", sp)
        DatasetCatalog.register(name, partial(_get_pairs, img_d, lbl_d))
        MetadataCatalog.get(name).set(
            stuff_classes=stuff_classes,
            stuff_colors=stuff_colors,
            ignore_label=IGNORE_VAL,
            evaluator_type="sem_seg",
        )

if __name__ == "__main__":
    register_clp_seg()
    print(f"Registered {len(SPLITS)} CLP segmentation splits, {len(stuff_classes)} classes.")
