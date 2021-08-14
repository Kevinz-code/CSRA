# CSRA
Official code of ICCV2021 paper. [Residual Attention: A Simple But Effective Method for Multi-Label Recoginition](https://arxiv.org/abs/2108.02456)<br>

### Training and Validation code has been released!
This package is developed by Mr. Ke Zhu (http://www.lamda.nju.edu.cn/zhuk/). If you have any problem about the code, please feel free to contact Mr. Ke Zhu (zhuk@lamda.nju.edu.cn). The package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Jianxin Wu (mail to wujx2001@gmail.com).

## Requirements
- Python 3.7
- pytorch 1.6
- torchvision 0.7.0
- pycocotools 2.0
- tqdm 4.49.0, pillow 7.2.0
$\lambda$

## Dataset
We expect VOC2007 and COCO2014 dataset to have the following structure
```
'''
root/
|-- train/
|   |-- class1 |-- image1.jpg 
|   |-- class2 |-- image2.jpg
|   |-- class3 |-- image3.jpg
|   ...
|-- val/
|   |-- class1 |-- image1.jpg 
|   |-- class2 |-- image2.jpg
|   |-- class3 |-- image3.jpg
|   ...
|-- myval/ (groundtruth annotation xml file, you can change the folder name, and modify it in Line 67 in PSOL_inference.py)
|   |-- class1 |-- image1.xml 
|   |-- class2 |-- image2.xml
|   |-- class3 |-- image3.xml
...
```


## Validation
We provide pretrained models for validation. Please refer to the following link to download them. 
|Dataset      | model     |  Head       |   mAP     | Download   |
|  ---------- | -------   |  ---------- | -------   | --------   |
| VOC2007     |ResNet-101 |     1       |  94.7     | Download   |
| VOC2007     |ResNet-cut |     1       |  95.2     | Download   |
| COCO        |ResNet-101 |     4       |  83.3     | Download   |
| COCO        |ResNet-101 |     6       |  83.5     | Download   |
| COCO        |ResNet-cut |     6       |  85.6     | Download   |

