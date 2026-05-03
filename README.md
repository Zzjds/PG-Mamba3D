# PG-Mamba3D

> Paper under review. Full technical details will be released upon acceptance.The current version is only the pfds version, serving as a preliminary test version. Its overall performance remains competitive.

<img width="446" height="443" alt="image" src="https://github.com/user-attachments/assets/30ff2739-546c-4f99-bda3-1207c1ac34be" />


## Overview

A dual-branch multimodal anomaly detection framework for industrial 
inspection, integrating RGB and depth modalities via a 
potential field-guided state space decoder.



## Requirements

- Python 3.8+
- PyTorch >= 1.13
- CUDA 11.6+

Install dependencies:
pip install -r requirements.txt

Key packages: mamba-ssm, einops, timm, scikit-learn, 
              tifffile, imageio, imgaug, pyzorder

## Dataset Preparation

### MVTec 3D-AD
Download from: https://www.mvtec.com/company/research/datasets/mvtec-3d-ad
Run preprocessing:
python utils/preprocess_mvtec3d.py --dataset_path ./data/mvtec3d

### Eyecandies
Download from: https://eyecandies.github.io/
Run preprocessing:
python utils/preprocess_eyecandies.py --dataset_path ./data/eyecandies

### DTD (anomaly source)
Download from: https://www.robots.ox.ac.uk/~vgg/data/dtd/
Place under: ./dtd/images/

Expected structure:
data/
├── mvtec3d/
│   ├── bagel/
│   │   ├── train/good/{rgb/, xyz/}
│   │   └── test/{good/, crack/, ...}/{rgb/, xyz/, gt/}
│   └── ...
└── eyecandies/
    ├── CandyCane/
    └── ...

## Training

Multimodal model (RGB + Depth):
python trainer.py \
    --dataset_type Mvtec3D_AD \
    --model_variant multimodal \
    --mode_type Fusion0 \
    --data_dir ./data/mvtec3d \
    --anomaly_source_path ./dtd/images \
    --epochs 700 --bs 4 --lr 1e-4

Unimodal model (RGB only):
python trainer.py \
    --dataset_type Mvtec3D_AD \
    --model_variant unimodal \
    --mode_type RGB \
    --data_dir ./data/mvtec3d \
    --anomaly_source_path ./dtd/images \
    --epochs 700 --bs 4 --lr 1e-4

## Project Structure

├── model/
│   ├── easynet_pfdt.py              # Unimodal model
│   └── easynet_pfdt_fusion_fixed.py # Multimodal model
├── data/
│   ├── mvtec3d_dataset.py
│   └── eyecandies_dataset.py
├── utils/
│   ├── loss.py
│   ├── perlin.py
│   ├── skew_gaussian.py
│   └── au_pro_util.py
├── trainer.py
└── README.md

## Acknowledgements

This codebase builds upon EasyNet and DAS3D.
We thank the authors for their open-source contributions.

## License

This project is released under the MIT License.
