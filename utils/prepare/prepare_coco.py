import os
import json
import argparse
import numpy as np
from pycocotools.coco import COCO



def make_data(data_path=None, tag="train"):
    annFile = os.path.join(data_path, "annotations/instances_{}2014.json".format(tag))
    coco = COCO(annFile)

    img_id = coco.getImgIds()
    cat_id = coco.getCatIds()
    img_id = list(sorted(img_id))
    cat_trans = {}
    for i in range(len(cat_id)):
        cat_trans[cat_id[i]] = i

    message = []


    for i in img_id:
        data = {}
        target = [0] * 80
        path = ""
        img_info = coco.loadImgs(i)[0]
        ann_ids = coco.getAnnIds(imgIds = i)
        anns = coco.loadAnns(ann_ids)
        if len(anns) == 0:
            continue
        else:
            for i in range(len(anns)):
                cls = anns[i]['category_id']
                cls = cat_trans[cls]
                target[cls] = 1
            path = img_info['file_name']
        data['target'] = target
        data['img_path'] = os.path.join(os.path.join(data_path, "images/{}2014/".format(tag)), path)
        message.append(data)

    with open('data/coco/{}_coco2014.json'.format(tag), 'w') as f:
        json.dump(message, f)
            


# The final json file include: train_coco2014.json & val_coco2014.json
# which is the following format:
# [item1, item2, item3, ......,]
# item1 = {
#      "target": 
#      "img_path":      
# }
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Usage: --data_path /your/dataset/path/COCO2014
    parser.add_argument("--data_path", default="Dataset/COCO2014/", type=str, help="The absolute path of COCO2014")
    args = parser.parse_args()

    if not os.path.exists("data/coco"):
        os.makedirs("data/coco")
    
    make_data(data_path=args.data_path, tag="train")
    make_data(data_path=args.data_path, tag="val")

    print("COCO data ready!")
    print("data/coco/train_coco2014.json, data/coco/val_coco2014.json")
