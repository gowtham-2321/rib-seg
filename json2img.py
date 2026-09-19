# json2img.py
#
# FIXED. Converts VinDr-RibCXR's polygon-annotation JSON into the
# per-rib binary PNG masks that utils/dataloader.py's UnetDataset
# expects (one subfolder per rib index, files named
# "<image_stem>_<RibLabel>.png"). The label list and the actual
# polygon-fill / mask-writing logic were already correct; the only
# problems were hardcoded paths that only existed on the original
# author's machine (annotations_base_path, images_base_path, the exact
# JSON filename, and the "val" output subfolder name) - all now CLI
# flags.
#
# Usage (once per split - train/val/test):
#   python json2img.py --annotations_json /path/to/Vindr_RibCXR_train_mask.json \
#       --images_base_path /path/to/Vxray --split train --output_dir /path/to/Vxray

import argparse
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# VinDr-RibCXR: R1-R10, L1-L10 (20 ribs). For CXRS's 24-rib annotation
# format, extend this list to R1-R12, L1-L12 and adjust the JSON
# parsing below to match that dataset's schema.
LIST_LABEL = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10',
              'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_json', type=str, required=True,
                         help='path to a VinDr-RibCXR *_mask.json file')
    parser.add_argument('--images_base_path', type=str, required=True,
                         help='base folder the JSON\'s "img" paths are relative to')
    parser.add_argument('--split', type=str, default='val', help='output subfolder name, e.g. train/val/test')
    parser.add_argument('--output_dir', type=str, required=True,
                         help='base output folder; masks are written to <output_dir>/<split>/<rib_index>/')
    args = parser.parse_args()

    data = pd.read_json(args.annotations_json)

    for i in range(len(data)):
        img_relative_path = data['img'][i]
        img_name = os.path.basename(img_relative_path)

        # FIX: the JSON's "img" field is a relative path (e.g.
        # "data/train/img/foo.png"), but the folder actually on disk
        # can differ in case (Kaggle's VinDr-RibCXR packaging uses
        # "Data" with a capital D) or layout - a case-sensitive
        # filesystem then fails a literal join. Try the path exactly
        # as the JSON gives it first (for a layout that matches), and
        # fall back to just the basename under images_base_path (for
        # when --images_base_path already points straight at the
        # split's img/ folder, which is the common case on Kaggle).
        candidates = [
            os.path.join(args.images_base_path, img_relative_path),
            os.path.join(args.images_base_path, img_name),
        ]
        img_path = next((p for p in candidates if os.path.exists(p)), None)
        if img_path is None:
            raise FileNotFoundError(
                f"Could not find image for '{img_relative_path}'. Tried:\n  "
                + "\n  ".join(candidates)
                + "\nCheck --images_base_path."
            )

        img = Image.open(img_path).convert('RGB')
        img = np.asarray(img, dtype=np.uint8)

        for index, label_name in enumerate(LIST_LABEL):
            label = np.zeros(img.shape[:2], dtype=np.uint8)

            pts = data[label_name][i]
            if pts != 'None':
                pts = np.array([[[int(pt['x']), int(pt['y'])]] for pt in pts])
                label = cv2.fillPoly(label, [pts], 255)

            binary_masks_dir = os.path.join(args.output_dir, args.split, str(index))
            os.makedirs(binary_masks_dir, exist_ok=True)

            binary_mask_path = os.path.join(
                binary_masks_dir, f'{os.path.splitext(img_name)[0]}_{label_name}.png'
            )
            cv2.imwrite(binary_mask_path, label)
            print(f'{binary_mask_path} saved')


if __name__ == "__main__":
    main()