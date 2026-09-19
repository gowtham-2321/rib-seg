# KAGGLE.md — running this on Kaggle

## 0. Push the fixes to your fork first

I patched these files locally: `networks/vit_seg_modeling.py`,
`networks/unet.py` (new), `nets/unet_training.py`, `utils/dataloader.py`,
`utils/utils_fit.py`, `train.py`, `predict.py`, `predict_allmetric.py`,
`predictunlabel.py`, `json2img.py`, `README.md`, plus `FIXES.md` and
this file. Commit and push all of them to
`https://github.com/gowtham-2321/rib-seg` before starting a Kaggle
notebook — the notebook will `git clone` your fork, so if the fixes
aren't pushed, Kaggle gets the same broken state described in
`FIXES.md`.

If you'd rather not push yet, zip the whole repo folder and upload it
as a private Kaggle Dataset instead (Kaggle -> Datasets -> New Dataset
-> upload the zip); it'll mount at
`/kaggle/input/<your-dataset-name>/`, and you `cp -r` it into
`/kaggle/working/rib-seg` as your first cell instead of `git clone`.

## 1. Notebook settings

New Notebook -> Settings (right sidebar):
- **Accelerator**: GPU (T4 x2 or P100 — either works; batch size may
  need lowering on a single smaller GPU)
- **Internet**: On (needed for `pip install`, `git clone`, and the ViT
  weight download)
- **Persistence**: Files only, or Variables & Files — either way,
  only `/kaggle/working/` survives between sessions/versions.

## 2. Get your dataset onto Kaggle

Apply for VinDr-RibCXR at vindr.ai/ribcxr (20 ribs) or get CXRS access
(24 ribs — check Huang et al. 2025's paper for a release link, since
it isn't linked from this repo). Once you have the raw images +
annotation JSON:
- Zip them and upload as a Kaggle Dataset (Datasets -> New Dataset),
  or
- Upload directly into this notebook's own Data panel.

Either way it mounts read-only at `/kaggle/input/<dataset-name>/`.

## 3. Setup cell

Paste `kaggle_setup.sh`'s contents into a cell (or run
`!bash kaggle_setup.sh` after uploading it), adjusting the ViT-weights
step if Google's bucket is down (see the comment in that file for
mirrors). This clones your fork and installs the ~3 packages Kaggle's
base image is missing (`ml-collections`, `Augmentor`, `straug`) — do
**not** run `pip install -r requirements.txt`, it targets an old
CUDA/torch combo plus heavy unused packages (detectron2/mmcv-full/
mmdet) that will fail to build here.

## 4. Convert annotations into per-rib masks

`/kaggle/input/` is read-only, so write the converted masks to
`/kaggle/working/`:

```
!python json2img.py \
    --annotations_json /kaggle/input/<dataset-name>/Vindr_RibCXR_train_mask.json \
    --images_base_path /kaggle/input/<dataset-name> \
    --split train --output_dir /kaggle/working/labels

!python json2img.py \
    --annotations_json /kaggle/input/<dataset-name>/Vindr_RibCXR_val_mask.json \
    --images_base_path /kaggle/input/<dataset-name> \
    --split val --output_dir /kaggle/working/labels
```

This writes `/kaggle/working/labels/train/<rib_index>/...png` and
`/kaggle/working/labels/val/<rib_index>/...png`.

## 5. Build train.txt / val.txt

One image name per line (no extension, no folder) — must match your
`.jpg` filenames under `/kaggle/input/<dataset-name>/images/`:

```python
import os
os.makedirs("/kaggle/working/splits", exist_ok=True)
img_dir = "/kaggle/input/<dataset-name>/images"
names = sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".jpg"))

# adjust this split however matches your dataset's official train/val division
n_val = max(1, len(names) // 10)
val_names, train_names = names[:n_val], names[n_val:]

with open("/kaggle/working/splits/train.txt", "w") as f:
    f.write("\n".join(train_names))
with open("/kaggle/working/splits/val.txt", "w") as f:
    f.write("\n".join(val_names))
```

## 6. Train

```
!python train.py --dataset vindr --backbone transunet --loss_fuc TPCloss \
    --data_path /kaggle/working/splits \
    --image_path /kaggle/input/<dataset-name>/images \
    --labels_path /kaggle/working/labels/train \
    --model_path model_data/imagenet21k+imagenet2012_R50+ViT-B_16.npz \
    --save_dir /kaggle/working/runs \
    --epochs 300 --batch_size 6 --num_workers 2
```

Swap `--dataset cxrs` for the 24-rib dataset, drop `--model_path` and
pass `--backbone unet` for the from-scratch UNet baseline instead of
TransUNet.

**Kaggle GPU sessions cap out around 9-12 hours.** 300 epochs may not
finish in one session. Either lower `--epochs` per run and resume by
pointing a later run's training loop at the saved checkpoint (this
version of `train.py` doesn't have a `--resume` flag — add one if you
need multi-session training, or just lower `--epochs` and accept a
shorter schedule), or use Kaggle's "Save Version" / "Save & Run All"
to keep `/kaggle/working/runs` between sessions and re-launch.

## 7. Evaluate

```
!python predict.py --backbone transunet --num_classes 20 \
    --model_path /kaggle/working/runs/.../best_epoch_weights.pth \
    --image_dir /kaggle/input/<dataset-name>/images_test \
    --pred_save_path /kaggle/working/predictions

!python predict_allmetric.py \
    --pred_dir /kaggle/working/predictions \
    --gt_dir /kaggle/working/labels/test \
    --num_classes 20 --out_txt /kaggle/working/results.txt
```

(Run `json2img.py --split test ...` first if you haven't already, to
get `/kaggle/working/labels/test/`.)

## 8. Get your results out

Anything under `/kaggle/working/` when you commit the notebook
("Save Version") becomes downloadable output. Checkpoints
(`*.pth`) can be large — consider only keeping
`best_epoch_weights.pth` per run if you're tight on Kaggle's output
size limits.
