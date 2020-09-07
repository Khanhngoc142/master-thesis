import os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from PIL import Image, ImageDraw
from extractor.crohme_parser.inkml import Ink
from utils import plt_draw, image_processing

if __name__ == "__main__":
    figside = 300
    mydpi = 96
    cur_path = os.path.abspath('.')
    print(cur_path)
    root_path = re.findall(r"^(.*hmer).*$", cur_path)[0]
    ink = Ink(os.path.join(root_path, "data/CROHME_full_v2/CROHME2013_data/TrainINKML/MfrDB/MfrDB0001.inkml"))

    group_coords = [g.trace_coords for g in ink.trace_groups]
    lbl_str = plt_draw.export_equation(group_coords, list("1+2"), "test_fig")
    print(lbl_str)
    with open("test_fig.label.txt", "w") as f:
        f.write(lbl_str)

    # fig, ax = plt_draw.plt_setup(figsize=(figside / mydpi, figside / mydpi), dpi=mydpi)
    # plt_draw.plt_draw_traces([t for g in group_coords for t in g], ax=ax)
    # plt_draw.plt_savefig("test_fig", None, dpi=mydpi)
    # label = '1 + 2'.split()
    # bboxes = [image_processing.get_trace_group_bbox(g) for g in group_coords]
    # xmin, ymin, xmax, ymax = zip(*bboxes)
    # width, height = fig.canvas.get_width_height()
    # xymin_pix = ax.transData.transform(np.vstack([xmin, ymin]).T)
    # xymax_pix = ax.transData.transform(np.vstack([xmax, ymax]).T)
    # xymin_pix = np.vstack([xymin_pix[:, 0], height - xymin_pix[:, 1]]).T.tolist()
    # xymax_pix = np.vstack([xymax_pix[:, 0], height - xymax_pix[:, 1]]).T.tolist()

    # new_bbox = [image_processing.np_get_trace_group_bbox(
    #     ax.transData.transform(np.vstack(list(zip([coord for t in g for coord in t]))).T)) for g in group_coords]

    # group_coords = [[coord for t in g for coord in t] for g in group_coords]

    # new_group_coords = [[np.vstack([t[0, :], height - t[1, :]]).T.tolist() for t in group] for group in new_group_coords]
    #
    # im = Image.open("test_fig.png")
    # draw = ImageDraw.Draw(im)
    #
    # for g in new_group_coords:
    #     for trace in g:
    #         for idx in range(len(trace) - 1):
    #             draw.line([(trace[idx][0], trace[idx][1]), (trace[idx + 1][0], trace[idx + 1][1])], fill="#ff0000",
    #                       width=3)
    #
    # im.show()

    # for bbx in new_bbox:
    #     draw.rectangle(bbx, width=3, outline="#ff0000")
    # im.show()

    # im = Image.open("test_fig.png")
    #
    # draw = ImageDraw.Draw(im)
    # new_bboxes = list(zip(xymin_pix, xymax_pix))

    # for bbx in new_bboxes:
    #     draw.rectangle([coord for xy in bbx for coord in xy], width=3, outline="#ff0000")
    # im.show()

    # img = mpimg.imread("test_fig.png")
    # fig, ax = plt.subplots()
    # _, ax = plt_draw.plt_setup()

    # ax.imshow(img)
    # plt_draw.plt_draw_bbox(new_bboxes[0])

    # coords = ink.trace_coords
    # ax = plt_draw_traces(coords)
    # plt.show()
