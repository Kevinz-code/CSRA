import os
import json
import random
import argparse


def make_wider(tag, value, data_path):
    img_path = os.path.join(data_path, "Image")
    ann_path = os.path.join(data_path, "Annotations")
    ann_file = os.path.join(ann_path, "wider_attribute_{}.json".format(tag))

    data = json.load(open(ann_file, "r"))

    final = []
    image_list = data['images']
    for image in image_list:
        for person in image["targets"]: # iterate over each person
            tmp = {}
            tmp['img_path'] = os.path.join(img_path, image['file_name'])
            tmp['bbox'] = person['bbox']
            attr = person["attribute"]
            for i, item in enumerate(attr):
                if item == -1:
                    attr[i] = 0
                if item == 0:
                    attr[i] = value  # pad un-specified samples
                if item == 1:
                    attr[i] = 1
            tmp["target"] = attr
            final.append(tmp)

    json.dump(final, open("data/wider/{}_wider.json".format(tag), "w"))
    print("data/wider/{}_wider.json".format(tag))



# which is the following format:
# [item1, item2, item3, ......,]
# item1 = {
#      "target": 
#      "img_path":      
# }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="Dataset/WIDER_ATTRIBUTE", type=str)
    args = parser.parse_args()

    if not os.path.exists("data/wider"):
        os.makedirs("data/wider")

    # 0 (zero) means negative, we treat un-specified attribute as negative in the trainval set
    make_wider(tag='trainval', value=0, data_path=args.data_path) 

    # 99 means we ignore un-specified attribute in the test set, following previous work
    # the number 99 can be properly identified when evaluating mAP
    make_wider(tag='test', value=99, data_path=args.data_path)
