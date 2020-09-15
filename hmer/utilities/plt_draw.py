import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plt_clear():
    plt.gcf().clear()


def plt_savefig(output_path, bbox_inches='tight', dpi=100, **kwargs):
    plt.savefig(output_path + '.png', bbox_inches=bbox_inches, dpi=dpi, **kwargs)


def plt_setup(**kwargs):
    """
    Prepare the canvas (figure and axis/axes) to draw
    :param kwargs: plt.subplots args
    :return: fig, ax
    """
    fig, ax = plt.subplots(**kwargs)
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine_type in ['left', 'right', 'top', 'bottom']:
        ax.spines[spine_type].set_visible(False)

    return fig, ax


def plt_trace_coords(coords, ax=None, linewidth=2, c='black'):
    if ax is None:
        fig, ax = plt_setup()
    ax.plot(*zip(*coords), linewidth=linewidth, c=c)
    return ax


def plt_draw_traces(traces, ax=None, linewidth=1, c='black'):
    """
    draw a list of traces
    :param traces: 3-layer list. list of list of coordinates
    :param ax:
    :param linewidth:
    :param c:
    :return:
    """
    for trace in traces:
        ax = plt_trace_coords(trace, ax=ax, linewidth=linewidth, c=c)
    return ax


def plt_draw_bbox(bbox, ax=None, scale=None, linewidth=2, color='r'):
    """
    Draw a bouding box using matplotlib.pyplot
    :param bbox: numpy array or list of list of coordinates (min_coord, max_coord)
    :param ax: pyplot ax
    :param scale: scale the bounding box by factor.
    :param linewidth: bouding box linewidth
    :param color: bouding box color
    :return: pyplot ax
    """
    if ax is None:
        ax = plt.gca()
    bbox = np.array(bbox).astype(np.float32)
    root_point = bbox[0]
    size = (bbox[1] - bbox[0])
    if scale is not None:
        assert isinstance(scale, (float, int)) \
               or (isinstance(scale, np.ndarray)
                   and (
                           scale.dtype == float or scale.dtype == int)), "Scale must be of type: int, float, numpy array of int or float"
        new_size = size * scale
        shift = (new_size - size) / 2
        root_point = root_point - shift
        size = new_size
    rect = patches.Rectangle(
        root_point,
        *size,
        linewidth=linewidth, edgecolor=color, facecolor='none'
    )
    ax.add_patch(rect)

    return ax



