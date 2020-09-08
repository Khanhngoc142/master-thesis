import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.utils.data as data
from utilities.fs import get_source_root


def parse_label_string(lbl_str):
    lbl = lbl_str.split()
    img = lbl[0]
    lbl = np.asarray(lbl[1:]).astype(np.float32).reshape((-1, 5)).tolist()
    lbl = [bbox[1:] + [int(bbox[0])] for bbox in lbl]
    return img, lbl


class CROHMEDetection4SSD(data.Dataset):
    def __init__(self, root, label_filename='labels.txt', transform=None):
        self.name = os.path.basename(root)
        self.root = root
        with open(os.path.join(root, label_filename)) as fin:
            data = fin.readlines()
        data = [parse_label_string(line) for line in data]
        data = pd.DataFrame(data, columns=['img', 'target'])
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root, self.data.iloc[idx, 0])
        img = cv2.imread(img_path)
        target = self.data.iloc[idx, 1]
        return torch.from_numpy(img), target


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utilities.plt_draw import plt_draw_bbox

    dataset = CROHMEDetection4SSD(root=os.path.join(get_source_root(), "demo-outputs/data/CROHME_2013_train"))

    # for chosen_idx in range(5):
    #     data = dataset[chosen_idx]
    #
    #     fig, ax = plt.subplots()
    #     ax.imshow(data[0])
    #
    #     def convert_target2bbox(target):
    #         return [[[t[0], t[1]], [t[2], t[3]]] for t in target]
    #
    #     for bbox in convert_target2bbox(data[1]):
    #         plt_draw_bbox(bbox, ax=ax)

