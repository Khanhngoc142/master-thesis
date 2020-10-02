import numpy as np
from utilities.metrics.simple_mAP import evaluate_AP


if __name__ == '__main__':
    # pred_path_aug = "/home/ubuntu/data/v_filter0.1_globalnms0.650.15_testssd0.10.1_modelvisdombest/aug_geotest.txt"
    pred_path_baseline = "/home/ubuntu/workspace/mine/master-thesis.git/hmer/testing/result_epoch30/test.txt"
    gt_path = "/home/ubuntu/workspace/mine/master-thesis.git/hmer/training/data/CROHME_2013_valid/labels_old.txt"
    # eval_dict_aug, pd_aug_info = evaluate_AP(gt_path, pred_path_aug)
    eval_dict_baseline, pd_base_info = evaluate_AP(gt_path, pred_path_baseline)
    # mAP_aug = np.array(list(eval_dict_aug.values())).mean()
    mAP_baseline = np.array(list(eval_dict_baseline.values())).mean()
    print(mAP_baseline)
