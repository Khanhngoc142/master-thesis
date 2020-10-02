from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
from extractor.crohme_parser.inkml import Ink
from utilities.plt_draw import plt_setup, plt_draw_traces
import os

if __name__ == "__main__":
    base = "/home/ubuntu/workspace/mine/master-thesis.git/hmer/data/CROHME_full_v2/CROHME2013_data/TestINKML/"
    files = os.listdir(base)
    size = 300
    dpi = 96

    # file_path = "/home/ubuntu/workspace/mine/master-thesis.git/hmer/data/CROHME_full_v2/CROHME2013_data/TestINKML/103_em_0.inkml"
    for file_name in files[:1]:
        print("Parsing "+ file_name)
        fig, ax = plt_setup(figsize=(size / dpi, size / dpi), dpi=dpi)
        file_path = base + file_name
        ink = Ink(file_path, is_test=True)
        plt_draw_traces(ink.trace_coords, ax=ax, linewidth=1)
        os.makedirs('/home/ubuntu/workspace/mine/master-thesis.git/hmer/testing/data2/', exist_ok=True)
        plt.savefig('/home/ubuntu/workspace/mine/master-thesis.git/hmer/testing/data2/' + file_name.split('.')[0] + '.png', dpi=dpi)

    # tree = ET.parse(file_path)
