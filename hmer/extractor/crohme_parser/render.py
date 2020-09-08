import numpy as np
import matplotlib.pyplot as plt
import os

from utils.data_processing import normalize_label
from utils.image_processing import get_trace_group_bbox
from utils.plt_draw import plt_clear, plt_setup, plt_draw_traces
from utils.fs import get_source_root


def render_equation(equation, label, output_path, size=300, dpi=96):
    """
    :param equation: 4-layer list equation groups' coordinates
    :param label: list form equation
    :param output_path: output path. equation drawing will be save to output_path + .png
    :param size: output size
    :param dpi: your system's dpi. use this link to find your system's dpi https://www.infobyip.com/detectmonitordpi.php
    :return: label string with the following format
    """
    plt_clear()
    fig, ax = plt_setup(figsize=(size/dpi, size/dpi), dpi=dpi)
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

    return output_path + ".png " + " ".join([f"{lbl:d} {bbx[0][0]} {bbx[0][1]} {bbx[1][0]} {bbx[1][1]}" for lbl, bbx in zip(norm_label, bboxes_pix)])


def render_equation_from_ink(ink, output_dir="demo-outputs", overwrite=False):
    """
    render ink to output directory
    :param ink:
    :param output_dir:
    :param overwrite:
    :return:
    """
    equation = [g.trace_coords for g in ink.trace_groups]
    label = ink.flatten_label
    filename = os.path.basename(ink.file_path)
    out_file = os.path.join(get_source_root(), output_dir, filename)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    if os.path.exists(out_file + '.png') and not overwrite:
        raise FileExistsError(out_file + '.png')
    label_str = render_equation(equation, label, out_file)
    with open(out_file + '.lbl.txt', 'w') as f:
        f.write(label_str)
