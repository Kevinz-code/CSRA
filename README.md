# CSRA 
This is the official code & website of ICCV 2021 paper: [Residual Attention: A Simple But Effective Method for Multi-Label Recoginition](https://arxiv.org/abs/2108.02456)<br>

![attention](https://github.com/Kevinz-code/CSRA/blob/master/utils/pipeline.PNG)

### Training and Validation code has been released!
This package is developed by Mr. Ke Zhu (http://www.lamda.nju.edu.cn/zhuk/). If you have any problem about the code, please feel free to contact Mr. Ke Zhu (zhuk@lamda.nju.edu.cn). The package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Jianxin Wu (mail to 
wujx2001@gmail.com).

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
which will automatically result in json files in *./data/voc07* and *./data/coco*

## Validation
We provide pretrained models for validation. ResNet101 trained on ImageNet with **CutMix** augmentation can be downloaded 
[here](https://drive.google.com/u/0/uc?export=download&confirm=kYfp&id=1T4AxsAO2tszvhn62KFN5kaknBtBZIpDV).
|Dataset      | Backbone  |   Head nums   |   mAP  |  Resolution     | Download   |
|  ---------- | -------   |  :--------:   | ------ |  :---:          | --------   |
| VOC2007     |ResNet-101 |     1         |  94.7  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=bXcv&id=1cQSRI_DWyKpLa0tvxltoH9rM4IZMIEWJ)   |
| VOC2007     |ResNet-cut |     1         |  95.2  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=otx_&id=1bzSsWhGG-zUNQRMB7rQCuPMqLZjnrzFh)  |
| COCO        |ResNet-101 |     4         |  83.3  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=EWtH&id=1e_WzdVgF_sQc--ubN-DRnGVbbJGSJEZa)   |
| COCO        |ResNet-cut |     6         |  85.6  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=uEcu&id=17FgLUe_vr5sJX6_TT-MPdP5TYYAcVEPF)   |

After model preparation, you can run the following validation command:
```shell
CUDA_VISIBLE_DEVICES=0 python val.py --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20  --load_from PRETRAINED_MODEL.pth
```

## Training
#### VOC2007
You can run either of these two lines below 
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20
CUDA_VISIBLE_DEVICES=0 python main.py --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20 --cutmix CutMix_ResNet101.pth
```
Note that the first command uses the Official ResNet-101 backbone while the second command uses the ResNet-101 pretrained on ImageNet with CutMix augmentation
[link](https://drive.google.com/u/0/uc?export=download&confirm=kYfp&id=1T4AxsAO2tszvhn62KFN5kaknBtBZIpDV) (which is supposed to gain better performance).

#### MS-COCO
run the ResNet-101 with 4 heads
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --num_heads 6 --lam 0.5 --dataset coco --num_cls 80
```
run the ResNet-101 (pretrained with CutMix) with 6 heads
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --num_heads 6 --lam 0.5 --dataset coco --num_cls 80
```
You can feel free to adjust the hyper-parameters such as number of attention heads (--num_heads), or the Lambda (--lam). Still, the default values of them in the above command are supposed to be the best.


## Acknowledgement

We thank Lin Sui (http://www.lamda.nju.edu.cn/suil/) for his initial contribution to this project.
