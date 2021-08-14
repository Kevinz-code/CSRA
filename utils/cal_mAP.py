import os
import numpy as np
import torch
import json


def json_map(cls_id, pred_json, ann_json, types):
    assert len(ann_json) == len(pred_json)
    num = len(ann_json)
    predict = np.zeros((num), dtype=np.float64)
    target = np.zeros((num), dtype=np.float64)

    for i in range(num):
        predict[i] = pred_json[i]["scores"][cls_id]
        target[i] = ann_json[i]["target"][cls_id]

    if types == 'wider':
        tmp = np.where(target != 99)[0]
        predict = predict[tmp]
        target = target[tmp]
        num = len(tmp)

    if types == 'voc07':
        tmp = np.where(target != 0)[0]
        predict = predict[tmp]
        target = target[tmp]
        neg_id = np.where(target == -1)[0]
        target[neg_id] = 0
        num = len(tmp)


    tmp = np.argsort(-predict)
    target = target[tmp]
    predict = predict[tmp]


    pre, obj = 0, 0
    for i in range(num):
        if target[i] == 1:
            obj += 1.0
            pre += obj / (i+1)
    pre /= obj
    return pre













