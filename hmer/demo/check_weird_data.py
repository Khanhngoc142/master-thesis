import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from extractor.crohme_parser.inkml import Ink
import utilities.plt_draw as plt_draw
from utilities.fs import get_path


def draw_ink_w_new_box(ink_path, chosen_indices, default_dot_pad=5000):
    ink = Ink(get_path(ink_path))
    coords = ink.trace_coords
    ax = plt_draw.plt_draw_traces(coords)
    for idx, box in enumerate([g.bbox for g in ink.trace_groups]):
        if idx in chosen_indices:
            plt_draw.plt_draw_bbox(box, ax=ax)
            w, h = box[1][0] - box[0][0], box[1][1] - box[0][1]
            if w == 0 and h == 0:
                box = [(box[0][0] - default_dot_pad/2, box[0][1] - default_dot_pad/2), (box[1][0] + default_dot_pad/2, box[1][1] + default_dot_pad/2)]
            elif w == 0:
                new_w = 0.1 * h
                box = [(box[0][0] - new_w / 2, box[0][1]), (box[1][0] + new_w / 2, box[1][1])]
            elif h == 0:
                new_h = 0.1 * w
                box = [(box[0][0], box[0][1] - new_h / 2), (box[1][0], box[1][1] + new_h / 2)]
            plt_draw.plt_draw_bbox(box, ax=ax, color='b')
    ax.set_title(ink_path)


if __name__ == "__main__":
    data = pd.read_csv(get_path("demo-outputs/weird.txt"), header=None, sep='\t')
    data.columns = ['img_path', 'box_idx', 'box_label', 'w', 'h']
    grouped = data.groupby('img_path')
    i = 0
    for img in grouped.groups.keys():
        draw_ink_w_new_box(img, grouped.get_group(img)['box_idx'].to_list())
        i += 1
        if i > 10:
            break
