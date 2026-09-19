# predict.py
#
# REWRITTEN. The original predict.py:
#   1) did `from networks.vit_seg_modeling import SCNet as ViT_seg` -
#      there is no `SCNet` class anywhere in this repo (only
#      `VisionTransformer`, which predictunlabel.py already used
#      correctly).
#   2) called `pred, overpred, nonover, predx = unet(image)` - a
#      4-output interface that matches no model in this repository.
#      Every backbone here returns a single [B, num_classes, H, W]
#      logits tensor.
#   3) also contained a large block of dead code reporting per-class
#      metrics for clavicles/scapulae/lungs/trachea/mediastinum -
#      structures that don't exist in either VinDr-RibCXR's or CXRS's
#      rib-only label sets. Removed; metric computation now lives in
#      predict_allmetric.py, decoupled from inference.
#
# Usage:
#   python predict.py --backbone transunet --num_classes 20 \
#       --model_path runs/.../best_epoch_weights.pth \
#       --image_dir /path/to/test/images \
#       --pred_save_path /path/to/write/predictions

import argparse
import os

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import VisionTransformer

try:
    from networks.unet import UNet
except ImportError:
    UNet = None


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
        if UNet is None:
            raise ImportError("networks/unet.py not found - add it before using --backbone unet")
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
    parser.add_argument('--model_path', type=str, required=True, help='path to a *_epoch_weights.pth checkpoint')
    parser.add_argument('--image_dir', type=str, required=True, help='folder of test images')
    parser.add_argument('--pred_save_path', type=str, required=True,
                         help='output root; predictions for class i are written to <pred_save_path>/<i>/')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args)
    state_dict = torch.load(args.model_path, map_location=device)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.eval().to(device)

    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    print('Running inference...')
    for img_name in tqdm(os.listdir(args.image_dir)):
        img_path = os.path.join(args.image_dir, img_name)
        image = cv2.imread(img_path, 0)
        image = cv2.resize(image, (args.img_size, args.img_size))
        image = np.expand_dims(image, -1).repeat(3, axis=-1)   # [H, W, 3]
        image = transform(image).unsqueeze(0).to(device)       # [1, 3, H, W]

        with torch.no_grad():
            logits = model(image)                # [1, num_classes, H, W]
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        for i in range(args.num_classes):
            out_dir = os.path.join(args.pred_save_path, str(i))
            os.makedirs(out_dir, exist_ok=True)
            mask = (probs[i] > 0.5).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(out_dir, img_name), mask)

    print('Done. Predictions saved to', args.pred_save_path)


if __name__ == "__main__":
    main()
