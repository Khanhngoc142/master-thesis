import os
import numpy as np
import pandas as pd
from utilities.fs import get_path
from utils.augmentations import jaccard_numpy
from utilities.data_processing import idx2symbols
from IPython.display import display

GT_FILEPATH = "demo-outputs/data/CROHME_2013_train/labels.txt"  # TODO:
PRED_FILEPATH = "demo-outputs/data/CROHME_2013_train/labels.txt"  # TODO:
IOU_THRESHOLD = 0.5


def load_file(file_path):
    if not os.path.exists(file_path):
        raise OSError("{} file not found!".format(file_path))

    gt_dict = {}
    with open(file_path, 'r') as fin:
        for line in fin:
            line = line.strip().split()

            gt_dict[line[0]] = line[1:]

    return gt_dict


def counter_increment(counter, key, num=1):
    if key not in counter.keys():
        counter[key] = num
    else:
        counter[key] += 1


def compute_second_precision(s):
    if s.iloc[1] > s.iloc[0]:
        return s.iloc[0]
    else:
        return np.nan


def compute_average_precision(conf, correct, num_gt):
    df = pd.DataFrame({'conf': conf, 'correct': correct}) \
        .sort_values('conf', ascending=False) \

    # compute precision/recall
    df.loc[:, 'precision'] = df.correct.expanding().agg(lambda s: s.mean())
    df.loc[:, 'recall'] = df.correct.expanding().agg(lambda s: s.sum()/num_gt)

    # compute interpolated precision
    new_index = pd.Index(np.linspace(0, 1, 11), name='recall')
    df = df.groupby('recall').agg({'precision': 'max'}).reindex(new_index)
    if np.isnan(df.iloc[-1].precision):
        df.iloc[-1].precision = 0
    df.loc[:, 'precision'] = df.precision[::-1].expanding().agg(max)[::-1]

    # interpolating precision
    tmp_df = df[::-1].rolling(2).apply(compute_second_precision)[::-1]
    tmp_df = tmp_df[tmp_df.precision > 0]
    df = pd.concat([df, tmp_df]) \
        .reset_index(drop=False) \
        .sort_values(by=['recall', 'precision'], ascending=[True, False])

    # compute AP
    df = df.groupby('precision').agg(lambda s: max(s) - min(s)).reset_index(drop=False)

    return (df.precision * df.recall).sum()


def evaluate_AP(ground_truth_file, prediction_file, iou_threshold=0.5):
    gt_dict = load_file(get_path(ground_truth_file))
    pred_dict = load_file(get_path(prediction_file))

    images_per_class = {}
    counter_per_class = {}
    pred_eval_result = []

    for file in gt_dict.keys():
        gt_data = np.array(gt_dict[file]).reshape((-1, 5))
        gt_labels = gt_data[:, 0].astype('int')
        gt_coords = gt_data[:, 1:].astype('float')

        pred_data = np.array(pred_dict[file]).reshape((-1, 6))
        pred_labels = pred_data[:, 0].astype('int')
        pred_conf = pred_data[:, 1].astype('float')
        pred_coords = pred_data[:, 2:].astype('float')

        # FOR DEV ONLY. TODO: Replace this with the above block
        # pred_data = np.array(pred_dict[file]).reshape((-1, 5))
        # pred_labels = pred_data[:, 0].astype('int')
        # pred_conf = np.ones(shape=(pred_data.shape[0],), dtype='float')
        # pred_coords = pred_data[:, 1:].astype('float')

        for i in set(gt_labels):
            counter_increment(images_per_class, i)

        for i in gt_labels:
            counter_increment(counter_per_class, i)

        for idx in range(pred_data.shape[0]):
            pred_box = pred_coords[idx]
            gt_boxes = gt_coords[gt_labels == pred_labels[idx]]
            iou = jaccard_numpy(gt_boxes, pred_box)
            pred_eval_result.append([pred_labels[idx], pred_conf[idx], (iou > iou_threshold).sum() > 0])

    pred_eval_result = pd.DataFrame(data=pred_eval_result, columns=['cls', 'conf', 'correct'])

    average_precision_per_class = {}

    for cls in sorted(counter_per_class.keys()):
        print("Class {} AP :".format(idx2symbols[cls]), end="\t")
        ap = compute_average_precision(
            pred_eval_result[pred_eval_result.cls == cls].conf,
            pred_eval_result[pred_eval_result.cls == cls].correct,
            counter_per_class[cls])
        print(ap)
        average_precision_per_class[cls] = ap

    return average_precision_per_class

