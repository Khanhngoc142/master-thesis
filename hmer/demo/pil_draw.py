import os

import matplotlib.pyplot as plt

from extractor.crohme_parser.inkml import Ink
from utils.image_processing import scale_traces
from utils.pil_draw import pil_draw_traces
from utils.fs import get_source_root

if __name__ == "__main__":
    root_path = get_source_root()
    ink = Ink(os.path.join(root_path, "data/CROHME_full_v2/CROHME2013_data/TrainINKML/MfrDB/MfrDB0001.inkml"))
    coords = ink.trace_coords
    # scale coords
    _, coords = scale_traces(coords, 300)
    img = pil_draw_traces(coords)
    plt.imshow(img, cmap='gray')
    plt.show()

