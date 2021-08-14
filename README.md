# CSRA
Official code of ICCV2021 paper. [Residual Attention: A Simple But Effective Method for Multi-Label Recoginition](https://arxiv.org/abs/2108.02456)<br>

#### Training and Validation code on VOC2007 and MS-COCO has been updated!

## Requirements
- Python 3.7
- pytorch 1.6
- torchvision 0.7.0
- pycocotools 2.0
- tqdm 4.49.0, pillow 7.2.0
$\lambda$

## Validation
We provide pretrained models for validation. Please refer to the following link to download them. 
|Dataset      | model     |  Head       |   mAP     | Download   |
|  ---------- | -------   |  ---------- | -------   | --------   |
| VOC2007     |ResNet-101 |     1       |  94.7     | Download   |
| VOC2007     |ResNet-cut |     1       |  95.2     | Download   |
| COCO        |ResNet-101 |     4       |  83.3     | Download   |
| COCO        |ResNet-101 |     6       |  83.5     | Download   |
| COCO        |ResNet-cut |     6       |  85.6     | Download   |

