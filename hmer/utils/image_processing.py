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
    scale = target_diff/diff

    # compute new x, y origins
    xorigin = (box_size[1] - xdiff * scale)/2
    yorigin = (box_size[1] - ydiff * scale)/2
    new_traces_data = []
    for trace in traces_data:
        coords = [[xorigin + (coord[0] - xmin) * scale, yorigin + (coord[1] - ymin) * scale]
                  for coord in trace]
        new_traces_data.append(coords)
    return scale, new_traces_data
