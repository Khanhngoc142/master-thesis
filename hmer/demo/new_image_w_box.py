import numpy as np
from utilities.fs import get_path
from utilities import plt_draw


def process_line(l):
    l = l.strip().split()
    img_path = l[0]
    l = np.array(l[1:]).reshape((-1, 5))
    cls_ = l[:, 0]
    coords_ = l[:, 1:]
    conf_ = np.ones(cls_.shape, dtype=np.float)
    return img_path, (cls_.astype('int'), conf_.astype('float'), coords_.astype('float'))


if __name__ == "__main__":
    label_file = get_path("training/data/CROHME_2013_valid/labels.txt")
    with open(label_file, 'r') as f:
        data = f.readlines()

    data = dict([process_line(line) for line in data])
    chosen_img = "training/data/CROHME_2013_valid/rit_42160_4.png"

    classes, conf, boxes = data[chosen_img]

    plt_draw.visualize_img_w_boxes(chosen_img, boxes, classes, conf, ncols=2)
