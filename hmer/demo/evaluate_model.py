import numpy as np
from utilities.metrics.simple_mAP import evaluate_AP


if __name__ == '__main__':
    pred_path_aug = "/home/ubuntu/data/ssd_final_result/aug_geo_multithreshnmstest.txt"
    pred_path_baseline = "/home/ubuntu/data/ssd_final_result/baseline_multithreshnmstest.txt"
    pred_path_extra = '/home/ubuntu/data/ssd_final_result/aug_geo_extra_data_test_model24_multithreshnmstest.txt'
    gt_path = "/home/ubuntu/workspace/mine/master-thesis.git/hmer/training/data/CROHME_2013_test/labels_old.txt"

    eval_dict_aug, pd_aug_info = evaluate_AP(gt_path, pred_path_aug)
    eval_dict_baseline, pd_base_info = evaluate_AP(gt_path, pred_path_baseline)
    eval_dict_extra, pd_extra_info = evaluate_AP(gt_path, pred_path_extra)

    mAP_aug = np.array(list(eval_dict_aug.values())).mean()
    mAP_baseline = np.array(list(eval_dict_baseline.values())).mean()
    mAP_extra = np.array(list(eval_dict_extra.values())).mean()
    print(mAP_baseline, mAP_aug, mAP_extra)
