import numpy as np
import matplotlib.pyplot as plt
import os

from utilities.data_processing import normalize_label
from utilities.image_processing import get_trace_group_bbox
from utilities.plt_draw import plt_clear, plt_setup, plt_draw_traces
from utilities.fs import get_source_root
from extractor.crohme_parser.extract import Extractor
from extractor.crohme_parser.augmentation import InkAugmentor


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
    plt_draw_traces([trace for group in equation for trace in group], ax=ax)
    plt.savefig(output_path + '.png', dpi=dpi)
    bboxes = [get_trace_group_bbox(group) for group in equation]
    xmin, ymin, xmax, ymax = zip(*bboxes)
    _, height = fig.canvas.get_width_height()
    xymin_pix = ax.transData.transform(np.vstack([xmin, ymin]).T)
    xymax_pix = ax.transData.transform(np.vstack([xmax, ymax]).T)
    xymin_pix = np.vstack([xymin_pix[:, 0], height - xymin_pix[:, 1]]).T.tolist()
    xymax_pix = np.vstack([xymax_pix[:, 0], height - xymax_pix[:, 1]]).T.tolist()

    bboxes_pix = list(zip(xymin_pix, xymax_pix))
    norm_label = normalize_label(label)
    plt.close()

    return (output_path + ".png ").replace(get_source_root(), "").lstrip('/') + " ".join(
        [f"{lbl:d} {bbx[0][0]} {bbx[0][1]} {bbx[1][0]} {bbx[1][1]}" for lbl, bbx in zip(norm_label, bboxes_pix)])


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
        filename = os.path.basename(ink.file_path) + '_' + str(i)
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
    filename = os.path.basename(ink.file_path)
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
                       datasets="train", output_dir=os.path.join("demo-outputs", "data"), overwrite=False, limit=None, treo_aug=True):
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

        labels.append(lbl_str)
        labels.extend(aug_lbl_str_lst)
        i += 1
        if limit is not None:
            if i > limit:
                break
    with open(os.path.join(output_dir, "labels.txt"), 'w') as f:
        f.write('\n'.join(labels))
