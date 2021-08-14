# CSRA
Official code of ICCV2021 paper. [Residual Attention: A Simple But Effective Method for Multi-Label Recoginition](https://arxiv.org/abs/2108.02456)<br>

#### Training and Validation code on VOC2007 and MS-COCO has been updated!

## Requirements
- Python 3.7
- pytorch 1.6
- torchvision 0.7.0
- pycocotools 2.0
- tqdm 4.49.0, pillow 7.2.0

## Validation
We provide pretrained models for validation. Please refer to the following link to download them. 
|Dataset| model       |  Head  | $\lambda$  |
|  ---------- | -------  |  ---------- | -------   |
|VOC2007| ResNet-101   |   68.4%      |  76   |
|MS-COCO| ResNet-cut |  63.4%      |  45   |
