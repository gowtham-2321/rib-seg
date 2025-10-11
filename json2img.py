import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os


list_label = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10',
              'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10']


annotations_base_path = 'path/to/Vxray/Annotations'
images_base_path = 'path/to/Vxray'

 
train_annotations_path = os.path.join(annotations_base_path, 'test/Vindr_RibCXR_val_mask.json')
train_data = pd.read_json(train_annotations_path)

 
for i in range(len(train_data)):
 
    img_relative_path = train_data['img'][i]  # 示例："some/path/to/image.png"
    img_name = os.path.basename(img_relative_path)


    img_path = os.path.join(images_base_path, img_relative_path)


    img = Image.open(img_path)
    img = img.convert('RGB')
    img = np.asarray(img, dtype=np.uint8)

 
    for index, label_name in enumerate(list_label):
 
        label = np.zeros(img.shape[:2], dtype=np.uint8)

   
        pts = train_data[label_name][i]
        if pts != 'None':
            pts = np.array([[[int(pt['x']), int(pt['y'])]] for pt in pts])
            label = cv2.fillPoly(label, [pts], 255)

 
        binary_masks_dir = os.path.join(images_base_path, 'val', f'{index}')
        os.makedirs(binary_masks_dir, exist_ok=True)

 
        binary_mask_path = os.path.join(binary_masks_dir, f'{os.path.splitext(img_name)[0]}_{label_name}.png')
        cv2.imwrite(binary_mask_path, label)

        print(f' {binary_mask_path} save')

