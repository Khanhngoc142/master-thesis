import os
import matplotlib.pyplot as plt

from extractor.crohme_parser.inkml import Ink
import utilities.plt_draw as plt_draw
from utilities.fs import get_source_root
from utilities.image_processing import get_trace_group_bbox_size

if __name__ == "__main__":
    chosen_idx = 31
    ink = Ink(os.path.join(get_source_root(), "data/CROHME_full_v2/CROHME2013_data/TrainINKML/expressmatch/93_Nina.inkml"))
    coords = ink.trace_coords
    ax = plt_draw.plt_draw_traces(coords)
    for idx, box in enumerate([g.bbox for g in ink.trace_groups]):
        if idx == chosen_idx:
            w, h = get_trace_group_bbox_size(box)
            if w == 0 and h == 0:
                pass
            elif w == 0:
                new_w = 0.1 * h
                box = [(box[0] - new_w/2)]
            plt_draw.plt_draw_bbox(box, ax=ax)
    plt.show()
