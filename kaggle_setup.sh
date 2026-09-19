#!/bin/bash
# Run each numbered section below as its own Kaggle notebook cell
# (prefix shell lines with "!"), or paste the whole thing into one
# cell. Requires: Kaggle notebook Settings -> Accelerator: GPU,
# Internet: On.
set -e

# ---------------------------------------------------------------
# 1. Get the code: clone YOUR fork (already has the fixes), not the
#    original XWei98/LTSeg (which has the bugs documented in FIXES.md
#    and will not run).
# ---------------------------------------------------------------
git clone https://github.com/gowtham-2321/rib-seg.git
cd rib-seg

# ---------------------------------------------------------------
# 2. Install only what's missing. requirements.txt targets
#    torch==1.8.0+cu111 plus detectron2/mmcv-full/mmdet, none of which
#    are actually needed to run train.py/predict.py, and which will
#    almost certainly fail to build against Kaggle's current CUDA/
#    Python image - do NOT `pip install -r requirements.txt` on
#    Kaggle. Kaggle's base image already has torch+CUDA, torchvision,
#    numpy, pandas, opencv, Pillow, scipy, matplotlib and tqdm. Traced
#    every import in this repo (grep -r "^import\|^from") and the only
#    things actually missing are:
pip install --no-deps ml-collections Augmentor straug

# Sanity check: these should both print with no ImportError.
python -c "from networks.vit_seg_modeling import VisionTransformer, CONFIGS; print('networks OK')"
python -c "from utils.dataloader import UnetDataset, unet_dataset_collate; print('utils OK')"

# ---------------------------------------------------------------
# 3. Get the ViT-R50 backbone weights train.py expects for
#    --model_path (skip this + pass --backbone unet instead if you'd
#    rather train the from-scratch UNet baseline).
#    NOTE: Google's official
#    https://storage.googleapis.com/vit_models/... bucket has been
#    reported down/expired as of early 2026. Try it first; if it
#    403s/404s, search "R50+ViT-B_16.npz" on Hugging Face or Kaggle
#    Datasets and add it as a Kaggle input instead, or fetch it from
#    the Beckschen/TransUNet repo's README, which links a mirror.
# ---------------------------------------------------------------
mkdir -p model_data
wget -O model_data/imagenet21k+imagenet2012_R50+ViT-B_16.npz \
    https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/R50+ViT-B_16.npz \
    || echo "Google's ViT weight bucket failed - see the note above for mirrors, or use --backbone unet instead."

echo "Setup done. Still needed before training:"
echo "  1) add your VinDr-RibCXR/CXRS dataset as a Kaggle Dataset input"
echo "  2) run json2img.py to turn its JSON annotations into per-rib masks"
echo "  3) build train.txt/val.txt split lists"
echo "See KAGGLE.md for the exact commands."
