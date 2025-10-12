## [Learning with Explicit Topological Priors for Chest X-Ray Rib Segmentation](https://papers.miccai.org/miccai-2025/paper/0313_paper.pdf) (MICCAI2025 Accepted)

Xiaowei Zhao, Chenglong Li, Jin Tang, and Chuanfu Li

## Introduction
Chest X-ray (CXR) image examination is a primary tool for assessing thoracic abnormalities. It is widely utilized for initial diagnosis and screening of diseases due to its cost-effectiveness and low radiation dose. Segmentation of ribs in CXR images (CXR rib segmentation) facilitates rapid determination of lesion types and locations, thereby alleviating the workload of medical professionals. Deep learning-based methods have achieved significant progress but still face some challenges in CXR rib segmentation, such as the occlusion challenge caused by artifacts and the interlace challenge caused by the spatial overlap of ribs. Therefore, it can be observed that the topological knowledge of ribs is crucial for CXR rib segmentation but neglected in existing methods, including the connectivity and interactivity of ribs. To address these challenges, we propose a novel learning framework that integrates explicit topological  priors into segmentation networks for precise CXR rib segmentation. In particular, we introduce two modules including the connectivity prior embedding module and the interactivity prior embedding module. These modules are designed to explicitly encode the continuity and interactivity of ribs into deep learning models for end-to-end training. Both modules are plug-and-play and can be integrated into various networks. We conduct extensive experiments on VinDr-RibCXR and CXRS datasets to evaluate the segmentation accuracy of each rib using multiple metrics. Evaluation and visual results show that our method exhibits strong adaptability, seamlessly integrating with diverse architectures and enhancing performance across various networks.
 
## Clone Repository
```shell
git clone https://github.com/XWei98/LTSeg.git
cd LTSeg/
```

## Prepare Datasets
You can refer to the  [https://vindr.ai/ribcxr](https://vindr.ai/ribcxr)](https://vindr.ai/ribcxr)

After applying for the dataset, label processing is performed through json2img.py

## Requirements
 
```shell
conda create -n tpl python=3.7.16
conda activate tpl
pip install -r requirements.txt
```

## Training
CUDA_VISIBLE_DEVICES=0 python train.py 
## Testing&Evaluation
python predict.py
## Visualization
python visrib.py
## Citation and Star
Please cite the following paper and star this project if you use this repository in your research. Thank you!
```
@InProceedings{ZhaXia_Learning_MICCAI2025,
        author = { Zhao, Xiaowei AND Li, Chenglong AND Tang, Jin AND Li, Chuanfu},
        title = { { Learning with Explicit Topological Priors for Chest X-ray Rib Segmentation } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
        year = {2025},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15975},
        month = {September},
        page = {300 -- 309}
}

```
 

