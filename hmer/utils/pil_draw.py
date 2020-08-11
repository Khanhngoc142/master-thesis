import numpy as np
from PIL import Image, ImageDraw


def _get_bbox(traces_data):
    coords = [coords for trace in traces_data for coords in trace]
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    return (min_coords[0], min_coords[1]), (max_coords[0], max_coords[1])


def pil_draw_traces(traces_data, img_size=None, linewidth=3, c=0.0):
    """

    :param traces_data: list of list of coordinates
    :param img_size:
    :param linewidth: linewidth
    :param c: color to draw line. default to be 0.0 which is black
    :return: numpy.ndarray
    """
    if img_size is None:
        (xmin, ymin), (xmax, ymax) = _get_bbox(traces_data)
        img_size = [int(round(max(xmax - xmin, ymax - ymin)))] * 2
    canvas = np.full(shape=img_size, fill_value=1.0-c, dtype=np.float32)

    for trace in traces_data:
        if len(trace) == 1:
            # SINGLE POINT
            x_coord, y_coord = trace[0]
            canvas[x_coord, y_coord] = c
        else:
            # A LINE
            for idx in range(len(trace) - 1):
                img = Image.fromarray(canvas)
                draw = ImageDraw.Draw(img)
                draw.line([(trace[idx][0], trace[idx][1]), (trace[idx+1][0], trace[idx+1][1])], fill=c, width=linewidth)
                canvas = np.array(img)

    return canvas
