# CSRA 
This is the official code of ICCV 2021 paper:<br>
[Residual Attention: A Simple But Effective Method for Multi-Label Recoginition](https://arxiv.org/abs/2108.02456)<br>

![attention](https://github.com/Kevinz-code/CSRA/blob/master/utils/pipeline.PNG)

### Demo, Train and Validation code have been released! (including VIT on Wider-Attribute)
This package is developed by Mr. Ke Zhu (http://www.lamda.nju.edu.cn/zhuk/) and we have just finished the implementation code of ViT models. If you have any question about the code, please feel free to contact Mr. Ke Zhu (zhuk@lamda.nju.edu.cn). The package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Jianxin Wu (mail to 
wujx2001@gmail.com).

## Requirements
- Python 3.7
- pytorch 1.6
- torchvision 0.7.0
- pycocotools 2.0
- tqdm 4.49.0, pillow 7.2.0

## Dataset
We expect VOC2007, COCO2014 and Wider-Attribute dataset to have the following structure:
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
......
|-- WIDER/
|---- Annotations/
|------ wider_attribute_test.json
|------ wider_attribute_trainval.json
|---- Image/
|------ train/
|------ val/
|------ test/
...
```
Then directly run the following command to generate json file (for implementation) of these datasets.
```shell
python utils/prepare/prepare_voc.py  --data_path  Dataset/VOCdevkit
python utils/prepare/prepare_coco.py --data_path  Dataset/COCO2014
python utils/prepare/prepare_wider.py --data_path Dataset/WIDER
```
which will automatically result in annotation json files in *./data/voc07*, *./data/coco* and *./data/wider*

## Demo
We provide prediction demos of our models. The demo images (picked from VCO2007) have already been put into *./utils/demo_images/*, you can simply run demo.py by using our CSRA models pretrained on VOC2007:
```shell
CUDA_VISIBLE_DEVICES=0 python demo.py --model resnet101 --num_heads 1 --lam 0.1 --dataset voc07 --load_from OUR_VOC_PRETRAINED.pth --img_dir utils/demo_images
```
which will output like this:
```shell
utils/demo_images/000001.jpg prediction: dog,person,
utils/demo_images/000004.jpg prediction: car,
utils/demo_images/000002.jpg prediction: train,
...
```


## Validation
We provide pretrained models on [Google Drive](https://www.google.com/drive/) for validation. ResNet101 trained on ImageNet with **CutMix** augmentation can be downloaded 
[here](https://drive.google.com/u/0/uc?export=download&confirm=kYfp&id=1T4AxsAO2tszvhn62KFN5kaknBtBZIpDV).
|Dataset      | Backbone  |   Head nums   |   mAP(%)  |  Resolution     | Download   |
|  ---------- | -------   |  :--------:   | ------ |  :---:          | --------   |
| VOC2007     |ResNet-101 |     1         |  94.7  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=bXcv&id=1cQSRI_DWyKpLa0tvxltoH9rM4IZMIEWJ)   |
| VOC2007     |ResNet-cut |     1         |  95.2  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=otx_&id=1bzSsWhGG-zUNQRMB7rQCuPMqLZjnrzFh)  |
| COCO        |ResNet-101 |     4         |  83.3  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=EWtH&id=1e_WzdVgF_sQc--ubN-DRnGVbbJGSJEZa)   |
| COCO        |ResNet-cut |     6         |  85.6  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=uEcu&id=17FgLUe_vr5sJX6_TT-MPdP5TYYAcVEPF)   |
| COCO        |VIT_L16_224 |     8      |  86.5  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=1Rmm&id=1TTzCpRadhYDwZSEow3OVdrh1TKezWHF_)|
| COCO        |VIT_L16_224* |     8     |  86.9  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=xpbJ&id=1zYE88pmWcZfcrdQsP8-9JMo4n_g5pO4l)|
| Wider       |VIT_B16_224|     1         |  89.0  |  224x224 |[download](https://drive.google.com/u/0/uc?id=1qkJgWQ2EOYri8ITLth_wgnR4kEsv0bfj&export=download)   |
| Wider       |VIT_L16_224|     1         |  90.2  |  224x224 |[download](https://drive.google.com/u/0/uc?id=1da8D7UP9cMCgKO0bb1gyRvVqYoZ3Wh7O&export=download)   |

For voc2007, run the following validation example:
```shell
CUDA_VISIBLE_DEVICES=0 python val.py --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20  --load_from MODEL.pth
```
For coco2014, run the following validation example:
```shell
CUDA_VISIBLE_DEVICES=0 python val.py --num_heads 4 --lam 0.5 --dataset coco --num_cls 80  --load_from MODEL.pth
```
For wider attribute with ViT models, run the following
```shell
CUDA_VISIBLE_DEVICES=0 python val.py --model vit_B16_224 --img_size 224 --num_heads 1 --lam 0.3 --dataset wider --num_cls 14  --load_from ViT_B16_MODEL.pth
CUDA_VISIBLE_DEVICES=0 python val.py --model vit_L16_224 --img_size 224 --num_heads 1 --lam 0.3 --dataset wider --num_cls 14  --load_from ViT_L16_MODEL.pth
```
To provide pretrained VIT models on Wider-Attribute dataset, we retrain them recently, which has a slightly different performance (~0.1%mAP) from what has been presented in our paper. The structure of the VIT models is the initial VIT version (**An image is worth 16x16 words: Transformers for image recognition at scale**, [link](https://arxiv.org/pdf/2010.11929.pdf)) and the implementation code of the VIT models is derived from [http://github.com/rwightman/pytorch-image-models/](http://github.com/rwightman/pytorch-image-models/).
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
CUDA_VISIBLE_DEVICES=0 python main.py --num_heads 6 --lam 0.4 --dataset coco --num_cls 80 --cutmix CutMix_ResNet101.pth
```
You can feel free to adjust the hyper-parameters such as number of attention heads (--num_heads), or the Lambda (--lam). Still, the default values of them in the above command are supposed to be the best.

#### Wider-Attribute
run the VIT_B16_224 with 1 heads
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --model vit_B16_224 --img_size 224 --num_heads 1 --lam 0.3 --dataset wider --num_cls 14
```
run the VIT_L16_224 with 1 heads
```shell
CUDA_VISIBLE_DEVICES=0,1 python main.py --model vit_L16_224 --img_size 224 --num_heads 1 --lam 0.3 --dataset wider --num_cls 14
```
Note that the VIT_L16_224 model consume larger GPU space, so we use 2 GPUs to train them.
## Notice
To avoid confusion, please note the **4 lines of code** in Figure 1 (in paper) is only used in **test** stage (without training), which is our motivation. When our model is end-to-end training and testing, **multi-head-attention** (H=1, H=2, H=4, etc.) is used with different T values. Also, when H=1 and T=infty, the implementation code of **multi-head-attention** is exactly the same with Figure 1.

We didn't use any new augmentation such as **Autoaugment, RandAugment** in our ResNet series models.

## Acknowledgement

We thank Lin Sui (http://www.lamda.nju.edu.cn/suil/) for his initial contribution to this project.
