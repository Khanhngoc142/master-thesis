import os
import re
import matplotlib.pyplot as plt

from extractor.crohme_parser.inkml import Ink
from utilities.plt_draw import plt_draw_traces

if __name__ == "__main__":
    cur_path = os.path.abspath('.')
    print(cur_path)
    root_path = re.findall(r"^(.*hmer).*$", cur_path)[0]
    ink = Ink(os.path.join(root_path, "data/CROHME_full_v2/CROHME2013_data/TrainINKML/MfrDB/MfrDB0001.inkml"))
    coords = ink.trace_coords
    ax = plt_draw_traces(coords)
    plt.show()
