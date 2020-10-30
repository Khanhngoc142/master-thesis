import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from utilities.fs import get_path
from math import ceil, floor
import cv2
from utilities.data_processing import idx2symbols


def plt_clear():
    plt.gcf().clear()


def plt_savefig(output_path, bbox_inches='tight', dpi=100, **kwargs):
    plt.savefig(output_path + '.png', bbox_inches=bbox_inches, dpi=dpi, **kwargs)


def plt_config_ax(ax, inverse=True, x=False, y=False, spines=False):
    if inverse:
        ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.set_visible(x)
    ax.yaxis.set_visible(y)
    if not spines:
        for spine_type in ['left', 'right', 'top', 'bottom']:
            ax.spines[spine_type].set_visible(False)
    return ax


def plt_setup(**kwargs):
    """
    Prepare the canvas (figure and axis/axes) to draw
    :param kwargs: plt.subplots args
    :return: fig, ax
    """
    fig, ax = plt.subplots(**kwargs)
    return fig, plt_config_ax(ax)


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
    bbox[0, :] = np.floor(bbox[0, :])
    bbox[1, :] = np.ceil(bbox[1, :])
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


def visualize_img_w_boxes(img_path, boxes, classes=None, score=None, ncols=3, figsize=3):
    num_boxes = len(boxes)
    img_path = get_path(img_path)
    img = cv2.imread(img_path)
    img = img[:, :, (2, 1, 0)]  # to rgb
    nrows = int(ceil(num_boxes/ncols))
    fig = plt.figure(figsize=(ncols * figsize, nrows * 2 * figsize))
    outer_grid = gridspec.GridSpec(2, 1, wspace=0, hspace=0.1)

    # image
    ax = plt.Subplot(fig, gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[1])[0])
    plt_config_ax(ax, spines=True)
    ax.imshow(img)
    fig.add_subplot(ax)

    ax = plt.Subplot(fig, gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[1])[1])
    plt_config_ax(ax, spines=True)
    ax.imshow(img)
    fig.add_subplot(ax)
    # plt_draw_bbox([[0, 0], [10, 10]])  # debug

    inner_grid = gridspec.GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer_grid[0], hspace=0.25)
    pad = 3
    for i in range(nrows * ncols):
        box_ax = plt.Subplot(fig, inner_grid[i])
        if i < len(boxes):
            xmin, ymin, xmax, ymax = boxes[i]
            plt_draw_bbox([[xmin, ymin], [xmax, ymax]], ax)
            xmin = int(floor(xmin)) - pad
            ymin = int(floor(ymin)) - pad
            xmax = int(ceil(xmax)) + pad
            ymax = int(ceil(ymax)) + pad
            box = img[ymin:ymax + 1, xmin:xmax + 1, :]
            plt_config_ax(box_ax, spines=True)
            plt.setp(box_ax.spines.values(), color='red')
            box_ax.imshow(box)
            if classes is not None:
                box_ax.set_title("Class {} with conf: {}".format(idx2symbols[classes[i]], score[i]), pad=0)
        else:
            plt_config_ax(box_ax)

        fig.add_subplot(box_ax)

    # plt.tight_layout()
    plt.show()
