import numpy as np
from utilities.fs import get_path
from utilities import plt_draw


def process_line(line, ground_truth=False):
    line = line.strip().split()
    img_path = line[0]
    if ground_truth:
        line = np.array(line[1:]).reshape((-1, 5))
        cls_ = line[:, 0]
        coords_ = line[:, 1:]
        conf_ = np.ones(cls_.shape, dtype=np.float)
    else:
        line = np.array(line[1:]).reshape((-1, 6))
        cls_ = line[:, 0]
        coords_ = line[:, 2:]
        conf_ = line[:, 1]
    return img_path, (cls_.astype('int'), conf_.astype('float'), coords_.astype('float'))


if __name__ == "__main__":
    label_file = get_path("/home/ubuntu/data/v_filter0.1_globalnms0.650.15_ssdtest0.10.1_aug_extra_data/aug_geo_extra_datatest.txt")
    with open(label_file, 'r') as f:
        data = f.readlines()

    data = dict([process_line(line, ground_truth=False) for line in data])
    chosen_img = "training/data/CROHME_2013_valid/122_em_347{}.png"

    classes, conf, boxes = data[chosen_img.format('')]

    plt_draw.visualize_img_w_boxes(chosen_img.format('.inkml'), boxes, classes, conf, ncols=4)
