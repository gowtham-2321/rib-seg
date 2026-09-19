# predictunlabel.py
#
# FIXED. Unlike predict.py / predict_allmetric.py, this script's model
# call was already correct (`pred = unet(image)`, a single tensor -
# it never imported the nonexistent `SCNet` class). Its problems were:
#   1) Hardcoded absolute paths that only existed on the original
#      author's machine (e.g. "/data1/Code/zhaoxiaowei/..."), including
#      a checkpoint path baked into the source.
#   2) The same `num_classes` bug seen elsewhere: hardcoded to 24
#      locally while an unrelated, unused `argparse --num_classes`
#      defaulted to 30.
# Both are fixed by exposing everything as CLI flags below. This
# script runs inference on external, unlabeled CXR datasets (its
# original use looks like it was checking domain generalization
# against JSRT / NIH / Shenzhen-style public chest X-ray datasets) -
# pass one or more --image_dirs.
#
# Usage:
#   python predictunlabel.py --backbone transunet --num_classes 20 \
#       --model_path runs/.../best_epoch_weights.pth \
#       --image_dirs /path/to/jsrt /path/to/nih --pred_save_path /path/to/output

import argparse
import os

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from networks.unet import UNet
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import VisionTransformer


def build_model(args):
    if args.backbone == "transunet":
        config_vit = CONFIGS_ViT_seg[args.vit_name]
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = args.n_skip
        if args.vit_name.find('R50') != -1:
            grid = int(args.img_size / args.vit_patches_size)
            config_vit.patches.grid = (grid, grid)
        return VisionTransformer(config_vit, img_size=args.img_size, num_classes=args.num_classes)
    elif args.backbone == "unet":
        return UNet(num_classes=args.num_classes)
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='transunet', choices=['transunet', 'unet'])
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16')
    parser.add_argument('--vit_patches_size', type=int, default=16)
    parser.add_argument('--n_skip', type=int, default=3)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_dirs', type=str, nargs='+', required=True,
                         help='one or more folders of unlabeled test images, e.g. JSRT / NIH / Shenzhen')
    parser.add_argument('--pred_save_path', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args)
    state_dict = torch.load(args.model_path, map_location=device)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.eval().to(device)

    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    for image_dir in args.image_dirs:
        dataset_name = os.path.basename(os.path.normpath(image_dir))
        out_dir = os.path.join(args.pred_save_path, dataset_name)
        print(f"Running inference on {image_dir} -> {out_dir}")

        for img_name in tqdm(os.listdir(image_dir)):
            img_path = os.path.join(image_dir, img_name)
            image = cv2.imread(img_path, 0)
            image = cv2.resize(image, (args.img_size, args.img_size))
            image = np.expand_dims(image, -1).repeat(3, axis=-1)
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(image)
                probs = torch.sigmoid(logits)[0].cpu().numpy()

            for i in range(args.num_classes):
                class_dir = os.path.join(out_dir, str(i))
                os.makedirs(class_dir, exist_ok=True)
                mask = (probs[i] > 0.5).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(class_dir, img_name), mask)

    print('Done. Predictions saved to', args.pred_save_path)


if __name__ == "__main__":
    main()
