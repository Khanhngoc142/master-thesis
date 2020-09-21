import numpy as np
from utilities.metrics.simple_mAP import evaluate_AP


if __name__ == '__main__':
    pred_path = "/home/ubuntu/workspace/mine/master-thesis.git/hmer/testing/eval_by_map/aug_geo__/test.txt"
    gt_path = "/home/ubuntu/workspace/mine/master-thesis.git/hmer/training/data/CROHME_2013_valid/labels.txt"
    eval_dict = evaluate_AP(gt_path, pred_path)
    mAP = np.array(list(eval_dict.values())).mean()
    print(mAP)
