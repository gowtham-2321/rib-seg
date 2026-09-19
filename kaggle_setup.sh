#!/bin/bash
# Kaggle setup. Run from the repo root, AFTER the notebook has cloned the repo
# and cd'd into it:   !bash kaggle_setup.sh
# Requires: Settings -> Accelerator: GPU, Internet: On.
set -e

# 1. Install only what's missing. Do NOT `pip install -r requirements.txt` on
#    Kaggle (it pins torch 1.8 / detectron2 / mmcv and will fail to build).
pip install --no-deps ml-collections Augmentor straug

# 2. OpenCV: the Distort augmentation needs the *contrib* build
#    (cv2.createThinPlateSplineShapeTransformer). Remove other OpenCV builds
#    first, since they all share the same `cv2` module.
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
pip install --no-deps opencv-contrib-python-headless

# 3. scikit-image pinned to the version this setup was tested with.
pip install --no-deps "scikit-image==0.24.0"

# 4. Patch straug for NumPy 2 / new scikit-image (see patch_straug.py).
python patch_straug.py

# 5. Sanity checks (fail loudly if something is off).
python -c "import cv2; assert hasattr(cv2, 'createThinPlateSplineShapeTransformer'); print('cv2', cv2.__version__, 'OK')"
python -c "from networks.vit_seg_modeling import VisionTransformer, CONFIGS; print('networks OK')"
python -c "from utils.dataloader import UnetDataset, unet_dataset_collate; print('utils OK')"

# 6. Pretrained R50+ViT-B_16 weights (needed for --backbone transunet).
#    If Google's bucket is down, download the file elsewhere, upload it as a
#    Kaggle Dataset, and point --model_path at it instead.
mkdir -p model_data
if [ ! -s model_data/R50+ViT-B_16.npz ]; then
    wget -O model_data/R50+ViT-B_16.npz \
        https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz \
        || { rm -f model_data/R50+ViT-B_16.npz; echo "WEIGHTS DOWNLOAD FAILED - see the note above"; }
fi
ls -lh model_data

echo "Setup done. Next: json2img.py (masks), the train/val split lists, then train.py."