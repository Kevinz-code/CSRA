import json
import numpy as np



def json_metric(score_json, target_json, num_classes, types):
    assert len(score_json) == len(target_json)
    scores = np.zeros((len(score_json), num_classes))
    targets = np.zeros((len(target_json), num_classes))
    for index in range(len(score_json)):
        scores[index] = score_json[index]["scores"]
        targets[index] = target_json[index]["target"]


    return metric(scores, targets, types)

def json_metric_top3(score_json, target_json, num_classes, types):
    assert len(score_json) == len(target_json)
    scores = np.zeros((len(score_json), num_classes))
    targets = np.zeros((len(target_json), num_classes))
    for index in range(len(score_json)):
        tmp = np.array(score_json[index]['scores'])
        idx = np.argsort(-tmp)
        idx_after_3 = idx[3:]
        tmp[idx_after_3] = 0.

        scores[index] = tmp
        # scores[index] = score_json[index]["scores"]
        targets[index] = target_json[index]["target"]

    return metric(scores, targets, types)


def metric(scores, targets, types):
    """
    :param scores: the output the model predict
    :param targets: the gt label
    :return: OP, OR, OF1, CP, CR, CF1
    calculate the Precision of every class by: TP/TP+FP i.e. TP/total predict
    calculate the Recall by: TP/total GT
    """
    num, num_class = scores.shape
    gt_num = np.zeros(num_class)
    tp_num = np.zeros(num_class)
    predict_num = np.zeros(num_class)


    for index in range(num_class):
        score = scores[:, index]
        target = targets[:, index]
        if types == 'wider':
            tmp = np.where(target == 99)[0]
            # score[tmp] = 0
            target[tmp] = 0
        
        if types == 'voc07':
            tmp = np.where(target != 0)[0]
            score = score[tmp]
            target = target[tmp]
            neg_id = np.where(target == -1)[0]
            target[neg_id] = 0


        gt_num[index] = np.sum(target == 1)
        predict_num[index] = np.sum(score >= 0.5)
        tp_num[index] = np.sum(target * (score >= 0.5))

    predict_num[predict_num == 0] = 1  # avoid dividing 0
    OP = np.sum(tp_num) / np.sum(predict_num)
    OR = np.sum(tp_num) / np.sum(gt_num)
    OF1 = (2 * OP * OR) / (OP + OR)

    #print(tp_num / predict_num)
    #print(tp_num / gt_num)
    CP = np.sum(tp_num / predict_num) / num_class
    CR = np.sum(tp_num / gt_num) / num_class
    CF1 = (2 * CP * CR) / (CP + CR)

    return OP, OR, OF1, CP, CR, CF1
