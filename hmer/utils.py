import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
                   and (scale.dtype == float or scale.dtype == int)), "Scale must be of type: int, float, numpy array of int or float"
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
