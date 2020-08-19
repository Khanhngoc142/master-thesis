import numpy as np


def get_equation_bbox(equation):
    """
    Get bounding box of equation
    :param equation: 4-layer list. represents an equation
    :return: xmin, ymin, xmax, ymax
    """
    coords = [coord for trace_group in equation for trace in trace_group for coord in trace]
    xmin, ymin = np.min(coords, axis=0)
    xmax, ymax = np.max(coords, axis=0)
    return xmin, ymin, xmax, ymax


def get_equation_bbox_size(equation):
    xmin, ymin, xmax, ymax = get_equation_bbox(equation)
    return xmax - xmin, ymax - ymin


def shift_equation(equation, xshift, yshift):
    return [[[[coord[0] + xshift, coord[1] + yshift] for coord in trace] for trace in trace_group] for trace_group in equation]


def scale_equation(equation, scale, new_xorigin=0, new_yorigin=0):
    """
    Scale the whole equation
    :param equation: 4-layer list. represents an equation
    :param scale: scale
    :param new_xorigin: new origin for equation
    :param new_yorigin: new origin for equation
    :return: 4-layer list
    """
    if scale == 1:
        return equation

    xmin, ymin, xmax, ymax = get_equation_bbox(equation)

    output = []
    for trace_group in equation:
        out_trace_group = []
        for trace in trace_group:
            coords = [[new_xorigin + (coord[0] - xmin) * scale, new_yorigin + (coord[1] - ymin) * scale]
                      for coord in trace]
            out_trace_group.append(coords)
        output.append(out_trace_group)
    return output


def get_trace_group_bbox(trace_group_coords):
    """
    Get bounding box of a set of a trace group coordinates
    :param trace_group_coords: 3-layer list. represents a trace group
    :return: xmin, ymin, xmax, ymax
    """
    coords = [coord for trace in trace_group_coords for coord in trace]
    xmin, ymin = np.min(coords, axis=0)
    xmax, ymax = np.max(coords, axis=0)
    return xmin, ymin, xmax, ymax


def get_trace_group_bbox_size(trace_group_coords):
    xmin, ymin, xmax, ymax = get_trace_group_bbox(trace_group_coords)
    return xmax - xmin, ymax - ymin


def shift_trace_group_coords(trace_group_coords, xshift=0, yshift=0):
    """
    shift coordinates of trace group by some amount
    :param trace_group_coords: 3-layer list. represents a trace group
    :param xshift:
    :param yshift:
    :return: 3-layer list
    """
    return [[[coord[0] + xshift, coord[1] + yshift] for coord in trace] for trace in trace_group_coords]


def compute_scale_trace_group_bbox(xmin, ymin, xmax, ymax, box_size):
    w, h = xmax - xmin, ymax - ymin
    largest_side = max(w, h)
    scale = box_size / largest_side
    return scale


def compute_scale2height_trace_group_bbox(ymin, ymax, height):
    h = ymax - ymin
    scale = height / h
    return scale


def compute_scale2width_trace_group_bbox(xmin, xmax, width):
    w = xmax - xmin
    scale = width / w
    return scale


def compute_scale_trace_group_coords(trace_group_data, box_size):
    """
    compute scale of a trace group
    :param trace_group_data: 3-layer list. represents a trace group
    :param box_size: int. length of one side of desired square box.
    :return: float
    """
    return compute_scale_trace_group_bbox(*get_trace_group_bbox(trace_group_data), box_size)


def centerize_trace_group_coords(trace_group_data):
    w, h = get_trace_group_bbox_size(trace_group_data)
    return [[[coord[0] - w / 2, coord[1] - h / 2] for coord in trace] for trace in trace_group_data]


def scale_trace_group(trace_group_data, box_size=1):
    """
    Scale coordinates of an equantion or a trace groups or a trace to given range
    :param trace_group_data: 3-layer list. represents a trace group.
    :param box_size: int. size of box with origin at 0
    :return: 3-layer list. a new trace_group_data
    """
    xmin, ymin, xmax, ymax = get_trace_group_bbox(trace_group_data)
    scale = compute_scale_trace_group_bbox(xmin, ymin, xmax, ymax, box_size)
    w, h = xmax - xmin, ymax - ymin

    # compute new x, y origins
    xorigin = (box_size - w * scale) / 2
    yorigin = (box_size - h * scale) / 2
    new_traces_data = []
    for trace in trace_group_data:
        coords = [[xorigin + (coord[0] - xmin) * scale, yorigin + (coord[1] - ymin) * scale]
                  for coord in trace]
        new_traces_data.append(coords)
    return scale, new_traces_data


def scale_trace_group_v2(trace_group_data, box_size=1):
    """
    Scale coordinates of an equantion or a trace groups or a trace to given range.
    v2: Scale ad stick symbol to the x, y axes.
    :param trace_group_data: 3-layer list. represents a trace group.
    :param box_size: int. size of box with origin at 0
    :return: 3-layer list. a new trace_group_data
    """
    xmin, ymin, xmax, ymax = get_trace_group_bbox(trace_group_data)
    scale = compute_scale_trace_group_bbox(xmin, ymin, xmax, ymax, box_size)

    # compute new x, y origins
    new_traces_data = []
    for trace in trace_group_data:
        coords = [[(coord[0] - xmin) * scale, (coord[1] - ymin) * scale]
                  for coord in trace]
        new_traces_data.append(coords)
    return scale, new_traces_data


def scale2height_trace_group(trace_group_data, height=1):
    xmin, ymin, xmax, ymax = get_trace_group_bbox(trace_group_data)
    scale = compute_scale2height_trace_group_bbox(ymin, ymax, height)

    new_traces_data = []
    for trace in trace_group_data:
        coords = [[(coord[0] - xmin) * scale, (coord[1] - ymin) * scale]
                  for coord in trace]
        new_traces_data.append(coords)
    return scale, new_traces_data


def scale2width_trace_group(trace_group_data, width=1):
    xmin, ymin, xmax, ymax = get_trace_group_bbox(trace_group_data)
    scale = compute_scale2height_trace_group_bbox(xmin, xmax, width)

    new_traces_data = []
    for trace in trace_group_data:
        coords = [[(coord[0] - xmin) * scale, (coord[1] - ymin) * scale]
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
        new_x = coord[0] * (
                np.sin((np.pi / 2.0 - np.deg2rad(alpha))) - coord[0] * np.sin(np.deg2rad(alpha)) / 1000000.0)
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
        new_y = coord[1] * (
                np.sin((np.pi / 2.0 - np.deg2rad(alpha))) - coord[1] * np.sin(np.deg2rad(alpha)) / 1000000.0)
        new_x = coord[0]
        new_trace_coord.append([new_x, new_y])

    return new_trace_coord


def local_shrink(i, trace_coord, alpha=10):
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


def local_perspective(i, trace_coord, alpha=10):
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


def local_scale(trace_coord, k=0.7):
    """

    :param trace_coord:
    :param k: scale factor
    :return:
    """
    return [[k * coord[0], k * coord[1]] for coord in trace_coord]


def geometric_local_model(trace_coord, random_model, i, alpha, beta):
    """

    :param random_model:
    :param i:
    :param alpha:
    :param beta:
    :param gamma:
    :param k:
    :return:
    """

    if random_model == 1:
        return local_sheer(i, trace_coord, alpha)
    elif random_model == 2:
        return local_shrink(i, trace_coord, alpha)
    elif random_model == 3:
        return local_shrink_rotation(i, trace_coord, alpha, beta)
    elif random_model == 4:
        return local_vertical_perspective(trace_coord, alpha)
    else:
        return local_perspective_rotation(i, trace_coord, alpha, beta)


def geometric_global_transform(trace_coord, random_model, i, alpha, beta, gamma, k):
    """

    :param trace_coord:
    :param random_model:
    :param i:
    :param alpha:
    :param beta:
    :param gamma:
    :param k:
    :return:
    """

    local_transformed_coord = geometric_local_model(trace_coord,random_model,i, alpha, beta)
    tmp_coord_1 = local_scale(local_transformed_coord,k)
    return local_rotation(tmp_coord_1, gamma)