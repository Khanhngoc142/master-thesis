import os

import matplotlib.pyplot as plt

from extractor.crohme_parser.inkml import Ink
from utilities.image_processing import *
from utilities.pil_draw import pil_draw_traces
from utilities.plt_draw import *
from utilities.fs import get_source_root
from extractor.crohme_parser import augmentation

if __name__ == "__main__":
    root_path = get_source_root()
    ink = Ink(
        os.path.join(root_path, "data/CROHME_full_v2/CROHME2013_data/TrainINKML/MfrDB/MfrDB0057.inkml"))  # 57,77,17,27
    t_coords = ink.trace_coords
    # scale coords
    _, t_coords = scale_trace_group(t_coords, 300)
    # new_coords = []
    # for trace_group in ink.trace_groups:
    # _, scaled_coords = scale_traces(trace_group.trace_coords)
    # fig = plt.figure()
    # plt_clear()
    # fig, ax = plt_setup()

    # fig, axes = plt.subplots(nrows=4)
    img1 = plt_draw_traces(t_coords)
    # axes[0].invert_yaxis()
    # axes[0].set_aspect('equal', adjustable='box')
    # fig.add_subplot(3, 2, 1)
    # plt.imshow(img1, cmap='gray')
    plt.show()
    for n in range(1, 15):
        # new_coords = []
        #
        # for trace_group in ink.trace_groups:
        #     i = np.random.uniform(0, 1)
        #     print(i)
        #     for trace in trace_group.traces:
        #         tmp_coords = local_shrink_rotation(1, trace.coords, 3,-10)
        #         # tmp_coords = local_rotation(tmp_coords, 6)
        #         # tmp_coords = local_scale(tmp_coords, 1.2)
        #         new_coords.append(tmp_coords)
        #
        # print("\n")
        # ax = axes[n]
        # ax.invert_yaxis()
        # ax.set_aspect('equal', adjustable='box')
        new_group_coords = augmentation.InkAugmentor.geometric_transform(ink)
        print("\n")
        # plt.show()
        new_coords = [trace for group in new_group_coords for trace in group]
        _, coords = scale_trace_group(new_coords, 300)
        img = plt_draw_traces(coords)
        # fig.add_subplot(3,2,n)
        # plt.imshow(img, cmap='gray')
        plt.show()

plt.show()
