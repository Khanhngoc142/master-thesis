import os
import matplotlib.pyplot as plt
from extractor.crohme_parser import library, inkml
from utilities.plt_draw import plt_draw_traces
from utilities.fs import get_source_root, save_object


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
    root_path = get_source_root()
    demo_lib = library.build_library()
    out_dir = os.path.join(root_path, "demo-outputs")
    save_object(demo_lib, os.path.join(root_path, "demo-outputs", "lib.pkl"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "symbols.txt"), "w+") as f:
        for k in sorted(demo_lib.symbols):
            f.write("{}\n".format(k))

    # draw_demo(3, 4, demo_lib, 'a')
    # ink = inkml.Ink(os.path.join(root_path, "data/CROHME_full_v2/CROHME2013_data/TrainINKML/MfrDB/MfrDB3539.inkml"))
    # lib = library.Library()
    # lib.extract_from_ink(ink, (0, 1))
    # ink.convert_to_img("demo-outputs/test", write_simplified_label=True, draw_bbox=True)
    # print(inkml.TraceGroup._gen_id('1:2:3:4:'))
