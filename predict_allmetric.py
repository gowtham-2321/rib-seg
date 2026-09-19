# predict_allmetric.py
#
# REWRITTEN. Had the same blocking bugs as predict.py (nonexistent
# `SCNet` import, 4-tuple unpack of the model's output), plus a large
# block of dead code reporting metrics broken down by clavicles,
# scapulae, lungs, trachea and mediastinum - anatomical structures
# that exist in neither VinDr-RibCXR's nor CXRS's rib-only label sets.
#
# Deliberately decoupled from inference: run predict.py first to
# produce <pred_dir>/<class>/<image>.png masks, then point this script
# at that folder plus a matching ground-truth folder (as produced by
# json2img.py) to get the mIoU/mDSC/mSen/mSpec/mHD/mASSD numbers
# reported in Table 1, per rib and averaged.
#
# Usage:
#   python predict_allmetric.py --pred_dir /path/to/predictions \
#       --gt_dir /path/to/labels --num_classes 20 --out_txt results.txt

import argparse
import glob
import os

import cv2
import numpy as np
from scipy.spatial.distance import cdist, directed_hausdorff

METRIC_NAMES = ["mIoU", "mDSC", "mSen", "mSpec", "mHD", "mASSD"]


def load_binary(path, size=448):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (size, size))
    return (img > 125).astype(np.uint8)


def per_image_metrics(pred, gt):
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    inter = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    iou = inter / union if union else 1.0
    denom = pred_b.sum() + gt_b.sum()
    dsc = 2 * inter / denom if denom else 1.0

    tp = inter
    tn = (~pred_b & ~gt_b).sum()
    fp = (pred_b & ~gt_b).sum()
    fn = (~pred_b & gt_b).sum()
    sens = tp / (tp + fn) if (tp + fn) else 1.0
    spec = tn / (tn + fp) if (tn + fp) else 1.0

    y_t, x_t = np.where(gt_b)
    y_p, x_p = np.where(pred_b)
    if y_t.size and y_p.size:
        pts_t = np.stack([y_t, x_t], axis=1)
        pts_p = np.stack([y_p, x_p], axis=1)
        hd = max(directed_hausdorff(pts_t, pts_p)[0], directed_hausdorff(pts_p, pts_t)[0])
        d = cdist(pts_t, pts_p)
        assd = (d.min(axis=0).mean() + d.min(axis=1).mean()) / 2
    else:
        hd, assd = 0.0, 0.0

    return np.array([iou, dsc, sens, spec, hd, assd])


def eval_class(pred_dir, gt_dir):
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"Missing prediction folder: {pred_dir}")
    names = sorted(os.listdir(pred_dir))
    totals = np.zeros(len(METRIC_NAMES))
    n = 0
    for name in names:
        stem = os.path.splitext(name)[0]
        gt_matches = glob.glob(os.path.join(gt_dir, stem + "*"))
        if not gt_matches:
            continue
        pred = load_binary(os.path.join(pred_dir, name))
        gt = load_binary(gt_matches[0])
        totals += per_image_metrics(pred, gt)
        n += 1
    if n == 0:
        raise RuntimeError(f"No matching (prediction, ground-truth) pairs found for {pred_dir} vs {gt_dir}")
    return totals / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True,
                         help='root folder with one <class_idx>/ subfolder per rib (output of predict.py)')
    parser.add_argument('--gt_dir', type=str, required=True,
                         help='root folder with one <class_idx>/ subfolder per rib (output of json2img.py)')
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--out_txt', type=str, default=None, help='optional path to save a text report')
    args = parser.parse_args()

    per_class = []
    lines = []
    for c in range(args.num_classes):
        m = eval_class(os.path.join(args.pred_dir, str(c)), os.path.join(args.gt_dir, str(c)))
        per_class.append(m)
        line = f"class {c}: " + ", ".join(f"{n}={v:.4f}" for n, v in zip(METRIC_NAMES, m))
        print(line)
        lines.append(line)

    mean_m = np.mean(per_class, axis=0)
    summary = "MEAN over all classes: " + ", ".join(f"{n}={v:.4f}" for n, v in zip(METRIC_NAMES, mean_m))
    print(summary)
    lines.append(summary)

    if args.out_txt:
        with open(args.out_txt, "w") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    main()
