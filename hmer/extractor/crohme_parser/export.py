import numpy as np
import matplotlib.pyplot as plt
import os

from utilities.data_processing import normalize_label, idx2symbols
from utilities.image_processing import get_trace_group_bbox, couple_bbox, decouple_bbox
from utilities.plt_draw import plt_clear, plt_setup, plt_draw_traces, plt_draw_bbox
from utilities.fs import get_source_root
from extractor.crohme_parser.extract import Extractor
from extractor.crohme_parser.augmentation import InkAugmentor


def pad_zero_size_bbox(box):
    box = couple_bbox(*box)
    w, h = box[1][0] - box[0][0], box[1][1] - box[0][1]
    if w == 0 and h == 0:
        return None
    elif w == 0:
        new_w = 0.1 * h
        box = [(box[0][0] - new_w / 2, box[0][1]), (box[1][0] + new_w / 2, box[1][1])]
    elif h == 0:
        new_h = 0.1 * w
        box = [(box[0][0], box[0][1] - new_h / 2), (box[1][0], box[1][1] + new_h / 2)]
    return box


def bboxes(args):
    pass


def export_equation(equation, label, output_path, size=300, dpi=96):
    """
    :param equation: 4-layer list equation groups' coordinates
    :param label: list form equation
    :param output_path: output path. equation drawing will be save to output_path + .png
    :param size: output size
    :param dpi: your system's dpi. use this link to find your system's dpi https://www.infobyip.com/detectmonitordpi.php
    :return: label string with the following format
    """
    fig, ax = plt_setup(figsize=(size / dpi, size / dpi), dpi=dpi)
    bboxes = [get_trace_group_bbox(group) for group in equation]
    plt_draw_traces([trace for group in equation for trace in group], ax=ax)
    norm_label = list(normalize_label(label))
    # for i, box in enumerate(bboxes):
    #     plt_draw_bbox(couple_bbox(*box), ax=ax)
    #     ax.text(box[0], box[1], idx2symbols[norm_label[i]])
    plt.savefig(output_path + '.png', dpi=dpi)
    # bboxes = [get_trace_group_bbox(group) for group in equation]

    # correcting box
    norm_label, bboxes = zip(
        *[(nrm_lbl, decouple_bbox(pad_zero_size_bbox(box))) for nrm_lbl, box in zip(norm_label, bboxes) if
          pad_zero_size_bbox(box) is not None])

    xmin, ymin, xmax, ymax = zip(*bboxes)
    _, height = fig.canvas.get_width_height()
    xymin_pix = ax.transData.transform(np.vstack([xmin, ymin]).T)
    xymax_pix = ax.transData.transform(np.vstack([xmax, ymax]).T)
    xymin_pix = np.vstack([xymin_pix[:, 0], height - xymin_pix[:, 1]]).T.tolist()
    xymax_pix = np.vstack([xymax_pix[:, 0], height - xymax_pix[:, 1]]).T.tolist()

    bboxes_pix = list(zip(xymin_pix, xymax_pix))

    plt.close()

    return (output_path + ".png ").replace(get_source_root(), "").lstrip('/') + " ".join(
        [f"{lbl:d} {bbx[0][0]} {bbx[0][1]} {bbx[1][0]} {bbx[1][1]}" for lbl, bbx in zip(norm_label, bboxes_pix)])


def gen_data_with_geoaugment(ink, parent_label, parent_out_file, overwrite=False, n_random=3):
    labels = []
    for i in range(n_random):
        aug_equation = InkAugmentor.geometric_transform(ink)
        out_file = parent_out_file + f'_{i}'
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        if os.path.exists(out_file + '.png') and not overwrite:
            raise FileExistsError(out_file + '.png')
        label_str = export_equation(aug_equation, parent_label, out_file)
        labels.append(label_str)

    return labels


def export_from_ink_with_geoaugment(ink, output_dir, overwrite=False, write_label=True, n_random=5):
    """
        render ink with geometric transform to output directory
        :param ink:
        :param output_dir:
        :param overwrite:
        :param write_label:
        :param n_random: the number of transforms
        :return:
        """

    labels = []
    for i in range(n_random):
        aug_equation = InkAugmentor.geometric_transform(ink)
        label = ink.flatten_label
        filename = os.path.basename(ink.file_path).split(".")[0] + '_' + str(i)
        out_file = os.path.join(output_dir, filename)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        if os.path.exists(out_file + '.png') and not overwrite:
            raise FileExistsError(out_file + '.png')
        label_str = export_equation(aug_equation, label.split(), out_file)
        labels.append(label_str)
    if write_label:
        with open(out_file + '.lbl.txt', 'w') as f:
            f.write('\n'.join(labels))
    return labels


def export_from_ink(ink, output_dir, overwrite=False, write_label=True):
    """
    render ink to output directory
    :param ink:
    :param output_dir:
    :param overwrite:
    :param write_label:
    :return:
    """
    equation = [g.trace_coords for g in ink.trace_groups]
    label = ink.flatten_label
    filename = os.path.basename(ink.file_path).split(".")[0]
    out_file = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    if os.path.exists(out_file + '.png') and not overwrite:
        raise FileExistsError(out_file + '.png')
    label_str = export_equation(equation, label.split(), out_file)
    if write_label:
        with open(out_file + '.lbl.txt', 'w') as f:
            f.write(label_str)
    return label_str


def export_crohme_data(data_versions='2013', crohme_package=os.path.join(get_source_root(), "data", "CROHME_full_v2"),
                       datasets="train", output_dir=os.path.join("demo-outputs", "data"), overwrite=False, limit=None,
                       treo_aug=True):
    output_dir = os.path.join(get_source_root(), output_dir, f"CROHME_{data_versions}_{datasets}")
    os.makedirs(output_dir, exist_ok=True)
    extractor = Extractor(data_versions, crohme_package)
    labels = []
    i = 0
    for ink in extractor.parse_inkmls_iterator(datasets=datasets):
        print("Exporting ink {}...".format(ink.file_path))
        lbl_str = export_from_ink(ink, output_dir, write_label=False, overwrite=overwrite)
        if treo_aug:
            aug_lbl_str_lst = export_from_ink_with_geoaugment(ink, output_dir, write_label=False, overwrite=overwrite)
        else:
            aug_lbl_str_lst = []

        labels.append(lbl_str)
        labels.extend(aug_lbl_str_lst)
        i += 1
        if limit is not None:
            if i > limit:
                break
    with open(os.path.join(output_dir, "labels.txt"), 'w') as f:
        f.write('\n'.join(labels))


def generate_extra_training_data(symbol_lib, data_versions='2013', datasets="train",
                                 output_dir=os.path.join("demo-outputs", "data"), geo_aug=False, n_loop=3000):
    output_dir = os.path.join(get_source_root(), output_dir, f"CROHME_{data_versions}_{datasets}_extra")
    os.makedirs(output_dir, exist_ok=True)
    labels = []

    for i in range(n_loop):
        label, ink = InkAugmentor.equation_generate(symbol_lib)
        print(label)
        filename = f'gen_{i}'
        out_file = os.path.join(output_dir, filename)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        if os.path.exists(out_file + '.png'):
            raise FileExistsError(out_file + '.png')

        label_str = export_equation(ink, label, out_file)
        labels.append(label_str)

        if geo_aug:
            aug_labels = gen_data_with_geoaugment(ink, label, out_file)
            labels.extend(aug_labels)

    with open(os.path.join(output_dir, "labels.txt"), 'w') as f:
        f.write('\n'.join(labels))


def get_box_size(xmin, ymin, xmax, ymax):
    """

    :param xmin:
    :param ymin:
    :param xmax:
    :param ymax:
    :return: width, height
    """
    return xmax - xmin, ymax - ymin


def find_weird_boxes(data_versions='2013', crohme_package=os.path.join(get_source_root(), "data", "CROHME_full_v2"),
                     datasets="train", output_dir=os.path.join("demo-outputs", "data"), overwrite=False, limit=None,
                     treo_aug=True):
    weird = []
    target_labels = {
        '.': 0,
        ',': 0,
        '\prime': 0,
        '-': 0
    }
    output_dir = os.path.join(get_source_root(), output_dir, f"CROHME_{data_versions}_{datasets}")
    os.makedirs(output_dir, exist_ok=True)
    extractor = Extractor(data_versions, crohme_package)
    for ink in extractor.parse_inkmls_iterator(datasets=datasets):
        # print("Exporting ink {}...".format(ink.file_path))
        equation = [g.trace_coords for g in ink.trace_groups]
        labels = ink.flatten_label.split()
        bboxes = [get_trace_group_bbox(group) for group in equation]

        labels, bboxes = zip(*[(nrm_lbl, pad_zero_size_bbox(box)) for nrm_lbl, box in zip(labels, bboxes) if
                               pad_zero_size_bbox(box) is not None])

        bboxes = [get_box_size(*decouple_bbox(box)) for box in bboxes]
        for idx, (w, h) in enumerate(bboxes):
            if labels[idx] in target_labels:
                target_labels[labels[idx]] += 1
            if w == 0 or h == 0:
                report = "{}\t{}\t{}\t{}\t{}".format(ink.file_path.replace(get_source_root(), "").lstrip("/"), idx,
                                                     labels[idx], w, h)
                # print(report)
                weird.append(report)

    print(target_labels)
    return weird
