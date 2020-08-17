import numpy as np


def scale_traces(traces_data, box_size=(0, 1)):
    """
    Scale coordinates of an equantion or a trace groups or a trace to given range
    :param traces_data: list of list of coordinates, which is a 3D list [[[x,y]]]
    :param box_size:
    :return: a new traces_data
    """
    if isinstance(box_size, (list, tuple)):
        assert len(box_size) == 2, "target box_size must be tuple/list of 2 int/float or a single float"
    else:
        assert isinstance(box_size, (float, int)), "target box_size must be tuple/list of 2 int/float or a single float"
        box_size = (0, box_size)
    # flatten into a list of coordinates
    flatten_traces_data = np.array([coord for trace in traces_data for coord in trace])
    xmin, ymin = np.min(flatten_traces_data, axis=0)
    xmax, ymax = np.max(flatten_traces_data, axis=0)

    # compute scale
    xdiff = xmax - xmin
    ydiff = ymax - ymin
    diff = max(xdiff, ydiff)
    target_diff = box_size[1] - box_size[0]
    scale = target_diff / diff

    # compute new x, y origins
    xorigin = (box_size[1] - xdiff * scale) / 2
    yorigin = (box_size[1] - ydiff * scale) / 2
    new_traces_data = []
    for trace in traces_data:
        coords = [[xorigin + (coord[0] - xmin) * scale, yorigin + (coord[1] - ymin) * scale]
                  for coord in trace]
        new_traces_data.append(coords)
    return scale, new_traces_data


def local_horizontal_sheer(trace_coord, alpha=10):
    """

    :param trace_coord: coords of one trace of a symbol
    :param alpha:
    :return: new trace_coord after sheering horizontally by alpha
    """
    new_trace_coord = []
    for coord in trace_coord:
        x = coord[0]
        y = coord[1]
        new_x = x + y * np.tan(np.deg2rad(alpha))
        new_y = y
        new_trace_coord.append([new_x, new_y])

    return new_trace_coord


def local_vertical_sheer(trace_coord, alpha=10):
    """

    :param trace_coord: coord info of one trace of a symbol
    :param alpha:
    :return: new trace_coord after sheering vertically by alpha
    """
    new_trace_coord = []
    for coord in trace_coord:
        new_trace_coord.append([coord[0], coord[1] + coord[0] * np.tan(np.deg2rad(alpha))])

    return new_trace_coord


def local_sheer(i, trace_coord, alpha=10):
    # i = np.random.uniform(0, 1)
    return local_horizontal_sheer(trace_coord, alpha) if i >= 0.5 else local_vertical_sheer(trace_coord,
                                                                                            alpha)


def local_vertical_shrink(trace_coord, alpha=10):
    """

    :param trace_coord: list of coord of a trace of a symbol
    :param alpha:
    :return:
    """
    new_trace_coord = []
    for coord in trace_coord:
        new_x = coord[0] * (np.sin((np.pi / 2.0 - np.deg2rad(alpha))) - coord[0] * np.sin(np.deg2rad(alpha)) / 1000000.0)
        new_y = coord[1]
        new_trace_coord.append([new_x, new_y])

    return new_trace_coord


def local_horizontal_shrink(trace_coord, alpha=10):
    """

    :param trace_coord: list of coord of a trace of a symbol
    :param alpha:
    :return:
    """
    new_trace_coord = []
    for coord in trace_coord:
        new_y = coord[1] * (np.sin((np.pi / 2.0 - np.deg2rad(alpha))) - coord[1] * np.sin(np.deg2rad(alpha)) / 1000000.0)
        new_x = coord[0]
        new_trace_coord.append([new_x, new_y])

    return new_trace_coord


def local_shrink(i,trace_coord, alpha=10):
    # i = np.random.uniform(0, 1)
    return local_vertical_shrink(trace_coord, alpha) if i >= 0.5 else local_horizontal_shrink(
        trace_coord, alpha)


def local_rotation(trace_coord, beta=10):
    """

    :param trace_coord: list of coord of all traces of a symbol
    :param beta: angel of rotation
    :return: new coord of a symbol after rotation
    """
    rad_beta = np.deg2rad(beta)
    return [[coord[0] * np.cos(rad_beta) + coord[1] * np.sin(rad_beta),
             coord[0] * np.sin(rad_beta) + coord[1] * np.cos(rad_beta)] for coord in trace_coord]


def local_shrink_rotation(i, trace_coord, alpha=10, beta=10):
    """

    :param trace_coord: list of coord of all traces of a symbol
    :param alpha: angel to shrink
    :param beta: angel to rotate
    :return: new coord of a symbol after shrink then rotate
    """
    shrinked_coords = local_shrink(i, trace_coord, alpha)
    return local_rotation(shrinked_coords, beta)


def local_vertical_perspective(trace_coord, alpha=10):
    """

    :param trace_coord:
    :param alpha:
    :return:
    """

    new_trace_group = []
    rad_alpha = np.deg2rad(alpha)
    for coord in trace_coord:
        new_x = (2 / 3.0) * (coord[0] + 50 * np.cos(np.deg2rad(4 * alpha * ((coord[0] - 50) / 1000000.0))))
        new_y = (2 / 3.0) * coord[1] * (
            np.sin((np.pi / 2.0 - rad_alpha) - coord[1] * np.sin(rad_alpha) / 1000000.0))

        new_trace_group.append([new_x, new_y])

    return new_trace_group


def local_horizontal_perspective(trace_coord, alpha=10):
    """

    :param trace_coord:
    :param alpha:
    :return:
    """

    new_trace_coord = []
    rad_alpha = np.deg2rad(alpha)
    for coord in trace_coord:
        new_y = (2 / 3.0) * (coord[1] + 50 * np.cos(np.deg2rad(4 * rad_alpha * ((coord[1] - 50) / 100000.0))))
        new_x = (2 / 3.0) * coord[0] * (
            np.sin((np.pi / 2.0 - rad_alpha) - coord[0] * np.sin(rad_alpha) / 100000.0))

        new_trace_coord.append([new_x, new_y])

    return new_trace_coord


def local_perspective(i,trace_coord, alpha=10):
    """

    :param trace_coord:
    :param alpha:
    :return:
    """
    # i = np.random.uniform(0, 1)
    return local_vertical_perspective(trace_coord, alpha) if i >= 0.5 else local_horizontal_perspective(
        trace_coord, alpha)


def local_perspective_rotation(i, trace_coord, alpha=10, beta=10):
    """

    :param trace_coord:
    :param alpha:
    :param beta:
    :return:
    """
    perspective_coords = local_perspective(1, trace_coord, alpha)
    return local_rotation(perspective_coords, beta)
