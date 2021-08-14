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

## Dataset
We expect VOC2007 and COCO2014 dataset to have the following structure:
```
Dataset/
|-- VOCdevkit/
|---- VOC2007/
|------ JPEGImages/
|------ Annotations/
|------ ImageSets/
......
|-- COCO2014/
|---- annotations/
|---- images/
|------ train2014/
|------ val2014/
...
```
Then directly run the following command to generate json file (for implementation) of these datasets.
```shell
python utils/prepare_voc.py  --data_path  Dataset/VOCdevkit
python utils/prepare_coco.py --data_path  Dataset/COCO2014
```
which will automatically result in json files in ./data/voc07 and ./data/coco

## Validation
We provide pretrained models for validation. Please refer to the following link to download them. ResNet101 trained with CutMix can be downloaded [here](https://drive.google.com/u/0/uc?export=download&confirm=kYfp&id=1T4AxsAO2tszvhn62KFN5kaknBtBZIpDV) 
|Dataset      | model     |  Head       |   mAP     | Download   |
|  ---------- | -------   |  ---------- | -------   | --------   |
| VOC2007     |ResNet-101 |     1       |  94.7     |    |
| VOC2007     |ResNet-cut |     1       |  95.2     |   |
| COCO        |ResNet-101 |     4       |  83.3     | [link](https://drive.google.com/u/0/uc?export=download&confirm=EWtH&id=1e_WzdVgF_sQc--ubN-DRnGVbbJGSJEZa)   |
| COCO        |ResNet-101 |     6       |  83.5     | Download   |
| COCO        |ResNet-cut |     6       |  85.6     | Download   |


