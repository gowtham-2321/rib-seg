# train.py
#
# FIXED / REWRITTEN, with Kaggle-friendly defaults. Bugs fixed vs. the
# original:
#   1) `from nets.unet_training import ...` now resolves (nets/ exists).
#   2) `elif args.backbone == "unet": model = UNet(n_classes=num_classes)`
#      referenced a class that was never imported anywhere in the repo.
#      Fixed - see networks/unet.py.
#   3) `num_classes` was hardcoded to 20 as a local variable while a
#      separate, unused `argparse --num_classes` defaulted to 24 - the
#      two never talked to each other. Now one `--num_classes` flag.
#   4) `show_config(...)` requires `save_dir` as its first positional
#      argument (see utils/utils.py) - the original call never passed
#      one, guaranteed TypeError. Fixed.
#   5) `save_dir='/save_dir'`, `data_path='/data_split_file'`,
#      `image_path='Dataset/images'` were hardcoded paths that don't
#      exist on Kaggle (or anywhere but the original author's
#      machine). Replaced with Kaggle-appropriate defaults
#      (/kaggle/working/..., /kaggle/input/...) that you override via
#      CLI flags either way.
#   6) Removed a batch of unused argparse flags copy-pasted from
#      TransUNet's original Synapse-dataset example script
#      (--root_path, --list_dir, --max_iterations, --n_gpu,
#      --deterministic, --base_lr, --seed) that had no effect on
#      anything in this file.
#
# Kaggle usage (see the accompanying notebook / KAGGLE.md):
#   !python train.py --dataset vindr --backbone transunet --loss_fuc TPCloss \
#       --data_path /kaggle/input/vindr-ribcxr-splits \
#       --image_path /kaggle/input/vindr-ribcxr/images \
#       --labels_path /kaggle/input/vindr-ribcxr/labels/train \
#       --model_path /kaggle/input/vit-r50-weights/imagenet21k+imagenet2012_R50+ViT-B_16.npz \
#       --save_dir /kaggle/working/runs

import argparse
import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import show_config
from utils.utils_fit import fit_one_epoch

try:
    from networks.unet import UNet
except ImportError:
    UNet = None

# dataset -> num_classes. VinDr-RibCXR: R1-R10, L1-L10 (20 ribs).
# CXRS: R1-R12, L1-L12 (24 ribs).
DATASET_NUM_CLASSES = {"vindr": 20, "cxrs": 24}


def build_argparser():
    parser = argparse.ArgumentParser()

    # ---- dataset / paths (Kaggle-friendly defaults, override as needed) ----
    parser.add_argument('--dataset', type=str, default='vindr', choices=list(DATASET_NUM_CLASSES.keys()))
    parser.add_argument('--num_classes', type=int, default=None,
                         help='override the dataset preset\'s rib count if needed')
    parser.add_argument('--data_path', type=str, default='/kaggle/input/data-split-file',
                         help='folder containing train.txt / val.txt split lists')
    parser.add_argument('--image_path', type=str, default='/kaggle/input/dataset/images',
                         help='folder of input images (files named <name>.jpg)')
    parser.add_argument('--labels_path', type=str, default=None,
                         help='folder of per-rib label masks (defaults to <image_path>/labels)')
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/runs',
                         help='must be under /kaggle/working - it\'s the only writable, persisted directory')

    # ---- model ----
    parser.add_argument('--backbone', type=str, default='transunet', choices=['transunet', 'unet'])
    parser.add_argument('--model_path', type=str, default='',
                         help='TransUNet only: path to the ImageNet-21k R50+ViT-B_16 .npz pretrained weights '
                              '(leave empty to train from scratch)')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
    parser.add_argument('--vit_patches_size', type=int, default=16)
    parser.add_argument('--n_skip', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=448)

    # ---- loss ----
    parser.add_argument('--loss_fuc', type=str, default='TPCloss',
                         choices=['BCEloss', 'Diceloss', 'LTSloss', 'TPCloss'])

    # ---- optimization ----
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--optimizer_type', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_decay_type', type=str, default='cos', choices=['cos', 'step'])
    parser.add_argument('--num_workers', type=int, default=2, help='Kaggle notebooks have limited CPU - 2-4 is safe')
    parser.add_argument('--gpu', type=str, default='0')

    return parser


def build_model(args, num_classes):
    if args.backbone == 'transunet':
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            grid = int(args.img_size / args.vit_patches_size)
            config_vit.patches.grid = (grid, grid)
        model = ViT_seg(config_vit, img_size=args.img_size, num_classes=num_classes)
        if args.model_path:
            model.load_from(weights=np.load(args.model_path))
        return model
    elif args.backbone == 'unet':
        if UNet is None:
            raise ImportError("networks/unet.py not found - add it before using --backbone unet")
        return UNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")


def main():
    args = build_argparser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cuda = torch.cuda.is_available()

    num_classes = args.num_classes or DATASET_NUM_CLASSES[args.dataset]
    labels_path = args.labels_path or os.path.join(args.image_path, "labels")
    input_shape = (args.img_size, args.img_size)

    model = build_model(args, num_classes)
    if args.backbone == 'unet':
        weights_init(model)

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M')
    save_dir = os.path.join(args.save_dir, f"{args.dataset}_{args.backbone}_{args.loss_fuc}_{time_str}")
    os.makedirs(save_dir, exist_ok=True)

    loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    model_train = model.train()
    if cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    with open(os.path.join(args.data_path, "train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(args.data_path, "val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train, num_val = len(train_lines), len(val_lines)

    # FIX: show_config requires save_dir as its first positional arg;
    # the original call never passed one.
    show_config(
        save_dir,
        dataset=args.dataset, num_classes=num_classes, backbone=args.backbone,
        model_path=args.model_path, input_shape=input_shape, Epochs=args.epochs,
        batch_size=args.batch_size, Init_lr=args.init_lr, optimizer_type=args.optimizer_type,
        momentum=args.momentum, lr_decay_type=args.lr_decay_type, save_dir=save_dir,
        num_workers=args.num_workers, num_train=num_train, num_val=num_val, loss_fuc=args.loss_fuc,
    )

    nbs = 16
    lr_limit_max = 1e-4 if args.optimizer_type == 'adam' else 1e-1
    lr_limit_min = 1e-4 if args.optimizer_type == 'adam' else 5e-4
    init_lr_fit = min(max(args.batch_size / nbs * args.init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(args.batch_size / nbs * args.init_lr * 0.01, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    optimizer = {
        'adam': optim.Adam(model.parameters(), init_lr_fit, betas=(args.momentum, 0.999),
                            weight_decay=args.weight_decay),
        'sgd': optim.SGD(model.parameters(), init_lr_fit, momentum=args.momentum, nesterov=True,
                          weight_decay=args.weight_decay),
    }[args.optimizer_type]

    lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, init_lr_fit, min_lr_fit, args.epochs)

    epoch_step = max(1, num_train // args.batch_size)
    epoch_step_val = max(1, num_val // args.batch_size)

    train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, args.image_path, labels_path)
    val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, args.image_path, labels_path)

    eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, args.image_path, save_dir, cuda)

    no_improve_count = 0
    for epoch in range(args.epochs):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        gen = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True,
                          collate_fn=unet_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              collate_fn=unet_dataset_collate)

        no_improve_count = fit_one_epoch(
            model_train, model, loss_history, eval_callback, optimizer, epoch,
            epoch_step, epoch_step_val, gen, gen_val, args.epochs, args.loss_fuc,
            num_classes, save_dir, no_improve_count,
        )


if __name__ == "__main__":
    main()
