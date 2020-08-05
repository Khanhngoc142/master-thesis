import os
import matplotlib.pyplot as plt
from extractor.crohme_parser import library, inkml
from utils.plt_draw import plt_draw_traces


def draw_demo(nrows, ncols, lib, symbol):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    for i in range(nrows * ncols):
        ax = axes[int(i / ncols)][i % ncols]
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        for spine_type in ['left', 'right', 'top', 'bottom']:
            ax.spines[spine_type].set_visible(False)
        _, traces = lib.get_traces_of_symbol(symbol)
        _ = plt_draw_traces(traces, ax=ax)
    return fig, axes


if __name__ == "__main__":
    demo_lib = library.build_library()
    os.makedirs("demo-outputs", exist_ok=True)
    with open(os.path.join("demo-outputs", "symbols.txt"), "w+") as f:
        for k in sorted(demo_lib.symbols):
            f.write("{}\n".format(k))

    draw_demo(3, 4, demo_lib, 'a')
    # ink = inkml.Ink("data/CROHME_full_v2/CROHME2013_data/TrainINKML/MfrDB/MfrDB3539.inkml")
    # lib = library.Library()
    # lib.extract_from_ink(ink, (0, 1))
    # ink.convert_to_img("demo-outputs/test", write_simplified_label=True, draw_bbox=True)
    # print(inkml.TraceGroup._gen_id('1:2:3:4:'))
