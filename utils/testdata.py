import json

def validata(mydirs, olddirs):
    my = json.load(open(mydirs, "r"))
    old = json.load(open(olddirs, "r"))

    assert len(my) == len(old)
    print(len(my))

    for i in range(len(my)):
        myitem = my[i]
        olditem = old[i]

        if myitem['target'] != olditem['target']:
            print("error")

        if myitem['img_path'] != myitem['img_path']:
            print("error")

        # print(myitem,"\n", olditem)
        # input()
    
validata("data/coco/val_coco2014.json", "../datasetAnns/coco2014val.json")