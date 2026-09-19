# FIXES.md — second-pass audit of gowtham-2321/rib-seg

You'd already patched `utils/utils_fit.py` (fixed the syntax error,
routed `"TPCloss"` to the same branch as `"LTSloss"`, added a
`trainDice_loss` reconstruction) and added `utils/TI_loss.py` and
`nets/`. Good progress — but auditing the rest of the repo, several of
the *same* blocking bugs from the original public release were still
present untouched, plus one new one your patch introduced. Full list
below, in order of how badly each one blocks things.

## Still-blocking bugs found and fixed this pass

1. **`networks/vit_seg_modeling.py` still imported
   `networks/pixlevel.py`, which doesn't exist.** This alone raises
   `ModuleNotFoundError` and blocks *every* script that touches the
   network — train.py, predict.py, predict_allmetric.py,
   predictunlabel.py — before anything else runs. Confirmed the
   import (and the `Vit.VisionTransformer`/`Reconstruct` import next
   to it) is dead code, never referenced elsewhere in the file, and
   removed both.

2. **`utils/dataloader.py`: `UnetDataset.__init__` never set
   `self.dataset_path`** (still commented out) — `AttributeError` on
   the first sample. Also still read labels from a hardcoded
   `/path/to/vinxray/labels`. Added a `labels_path` constructor arg
   (defaults to `<dataset_path>/labels`) and look label files up by
   filename prefix via `glob` (matches what `json2img.py` actually
   writes: `<image_stem>_<RibLabel>.png`).

3. **`utils/dataloader.py`: `unet_dataset_collate` still unpacked a
   3-tuple** (`for img, png, sampng in batch`) against a dataset that
   only ever returns 2 values — `ValueError` on the first batch.
   Fixed.

4. **`predict.py` and `predict_allmetric.py` were untouched from the
   original release** — both still import a `SCNet` class that
   doesn't exist anywhere in the repo, and both still unpack the
   model's output as a 4-tuple (`pred, overpred, nonover, predx =
   unet(image)`) that no model here produces (every backbone returns
   one `[B, C, H, W]` tensor). Rewrote both around the real
   single-tensor interface, and dropped a block of dead code in both
   reporting metrics for clavicles/scapulae/lungs/trachea/mediastinum
   — structures that don't exist in this task's label set.

5. **`train.py` was untouched from the original release**:
   - `show_config(...)` was called without `save_dir`, which
     `utils/utils.py`'s `show_config(save_dir, **kwargs)` requires as
     its first positional arg — guaranteed `TypeError`.
   - `num_classes` was hardcoded to `20` locally while a separate,
     unused `argparse --num_classes` defaulted to `24` — the two
     never agreed with each other.
   - `elif args.backbone == "unet": model = UNet(n_classes=num_classes)`
     referenced a class that was never imported and doesn't exist
     anywhere in the repo (only `networks/unet.py`, added now).
   - `save_dir='/save_dir'`, `data_path='/data_split_file'`,
     `image_path='Dataset/images'` were hardcoded, non-Kaggle paths.
   - A batch of unused argparse flags left over from TransUNet's
     original Synapse-dataset example script (`--root_path`,
     `--list_dir`, `--max_iterations`, `--n_gpu`, `--deterministic`,
     `--base_lr`, `--seed`) had no effect on anything and were removed.
   Rewrote with Kaggle-friendly path defaults (`/kaggle/working/...`,
   `/kaggle/input/...`), all still overridable via CLI.

## New bug found in your patch (not present, or present differently, in the original)

6. **`nets/unet_training.py`'s `CE_Loss`/`Dice_loss` are the
   single-label/softmax versions** (for one-class-per-pixel tasks,
   e.g. VOC-style segmentation) - but this task's ground truth is
   multi-label: `utils/dataloader.py` stacks one independent binary
   mask per rib into `[B, C, H, W]`, and ribs *overlap* in the 2D
   projection (that's the entire reason the paper's Interactivity
   module exists). Traced the actual shapes:
   - `CE_Loss`'s `nt, ht, wt = target.size()` crashes outright
     (`ValueError: too many values to unpack`) on a 4-D target.
   - `Dice_loss`'s `nt, ht, wt, ct = target.size()` doesn't crash
     immediately — it just silently mis-reads a `[B, C, H, W]` tensor's
     axes as if it were `[B, H, W, C]` — and a few lines later this
     produces a `.view()` reshape that doesn't line up with the
     prediction tensor's shape, raising a broadcast-shape
     `RuntimeError` in the `tp`/`fp`/`fn` sums.
   - Your new `trainDice_loss` in `utils/utils_fit.py` (used by the
     default `"TPCloss"` path — this is the loss the repo trains with
     unless you override it) has the **exact same bug**: the mis-read
     `ht`/`wt` make `h != ht` spuriously true, which triggers a bogus
     `F.interpolate()` call that silently corrupts the prediction's
     spatial shape (squishing it toward the mis-read `ct`/`ht` values)
     before the same kind of broadcast crash further down.

   Fixed all three to treat every channel as an independent binary
   mask (sigmoid per channel, no cross-channel softmax, no bogus
   reshape/interpolate) — `CE_Loss` is now `BCEWithLogitsLoss`,
   `Dice_loss` and `trainDice_loss` now sum `tp`/`fp`/`fn` per channel
   over the batch+spatial dims directly, since inputs/target/
   criticals_map are all already `[B, C, H, W]` and already spatially
   aligned - no permute or interpolate needed.

   This means: as uploaded, the repo's *default* training
   configuration (`loss_fuc="TPCloss"`) would not have produced wrong
   numbers — it would have crashed on the first training step, after
   silently corrupting a tensor shape first. `weights_init`,
   `get_lr_scheduler`, and `set_optimizer_lr` in the same file were
   already fine and untouched.

7. **`utils/utils_fit.py`'s connectivity/interactivity map computation
   was still hardcoded to exactly 20 channels** (`for i in range(20)`,
   `.expand(-1, 20, -1, -1)` ×2), even though `num_classes` is already
   a parameter of `fit_one_epoch`. This silently only processed 20 of
   CXRS's 24 ribs. Generalized to use `num_classes`.

## Everything else

Your `utils/TI_loss.py` (a genuine, complete implementation of Gupta
et al.'s Topological Interaction loss) is unused dead code currently —
nothing imports it except `utils/utils_fit.py`'s
`from utils.TI_loss import TI_Loss`, and `TI_Loss` itself is never
called. Left it in place as-is since it's harmless and might be worth
wiring in later as an alternative to the paper's own CEM/IEM-style
critical-pixel map.

Still out of scope, same as any pass at this repo: the other 5
baselines from Table 1 (UNeXt/UNet++/AttentionUNet/UCTransNet/VM-UNet/
MedSAM), the datasets themselves, and pretrained checkpoints.

See KAGGLE.md for how to actually run this on Kaggle now that it works.
