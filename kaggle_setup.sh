#!/bin/bash
# Run this as a single Kaggle notebook cell (with "!" prefix per line, or
# save as a .sh file and run with `!bash kaggle_setup.sh`).
# Requires: Kaggle notebook Settings -> Accelerator: GPU, Internet: On.
set -e

# 1. Clone the paper's repo
git clone https://github.com/XWei98/LTSeg.git
cd LTSeg

# 2. Drop in the two files that are missing from the public release.
#    (Upload nets/unet_training.py, nets/__init__.py, utils/TI_loss.py, and
#    the patched utils/utils_fit.py to your Kaggle notebook's /kaggle/input
#    or /kaggle/working, then copy them in here. Example if you uploaded a
#    "ltseg-patch" dataset:)
mkdir -p nets
cp /kaggle/input/ltseg-patch/nets/unet_training.py nets/unet_training.py
cp /kaggle/input/ltseg-patch/nets/__init__.py nets/__init__.py
cp /kaggle/input/ltseg-patch/utils/TI_loss.py utils/TI_loss.py
cp /kaggle/input/ltseg-patch/utils/utils_fit.py utils/utils_fit.py   # overwrites the broken original

# 3. Trimmed dependency install -- the repo's own requirements.txt targets
#    torch==1.8.0 + CUDA 11.1, detectron2, mmcv-full, mmdet, etc. which are
#    NOT needed to run train.py/predict.py and will fail to build on
#    Kaggle's current image. Kaggle already ships a recent torch+CUDA build;
#    we just add the extra libraries the code actually imports.
pip install --no-deps \
    einops \
    ml-collections \
    Augmentor \
    straug \
    medpy \
    tensorboardX \
    scikit-image

# 4. Get the ViT-R50 backbone weights train.py expects at
#    model_data/imagenet21k+imagenet2012_R50+ViT-B_16.npz
#    NOTE: Google's official https://storage.googleapis.com/vit_models/...
#    bucket has been reported down/expired as of early 2026. Try it first;
#    if it 403s/404s, get the same file from the Beckschen/TransUNet repo's
#    README, which links a mirror, or from a community mirror (e.g. search
#    "R50+ViT-B_16.npz" on Hugging Face or Kaggle Datasets and upload it as
#    a Kaggle input instead).
mkdir -p model_data
wget -O model_data/imagenet21k+imagenet2012_R50+ViT-B_16.npz \
    https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/R50+ViT-B_16.npz \
    || echo "Google's ViT weight bucket failed -- see the note above for mirrors."

echo "Setup done. Remember you STILL need to:"
echo "  1) fix save_dir / data_path at the top of train.py to Kaggle-writable paths"
echo "  2) get VinDr-RibCXR or CXRS and build train.txt / val.txt (see json2img.py)"
echo "before running: python train.py"
