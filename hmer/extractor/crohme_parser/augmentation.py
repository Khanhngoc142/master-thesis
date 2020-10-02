import os
import random
import extractor.crohme_parser.inkml as inkml
from constants import simplified_lbl_sep
from utilities import plt_draw
from utilities import image_processing
import numpy as np


class InkAugmentor(object):
    @staticmethod
    def simple_augment(ink, n):
        trace_groups = ink.trace_groups
        for _ in range(n):
            while True:
                indices = sorted([random.randint(0, len(trace_groups) - 1) for _ in range(2)])
                if indices[1] - indices[0] > 1:
                    break
            aug_trace_groups = trace_groups[indices[0]:indices[1]]
            label = simplified_lbl_sep.join([
                group.label
                for group in aug_trace_groups
            ])
            yield label, aug_trace_groups

    @staticmethod
    def elastic_augment(img):
        """
        Elastic augment an image
        :param img: input image of type. 2d or 3d np.ndarray
        :return: augmented image
        """
        # TODO: implement this please
        return img

    @staticmethod
    def geometric_transform(ink):
        """

        :param ink:
        :return:
        """
        new_group_coords = []
        gamma = random.randint(-10, 10)
        k = round(random.uniform(0.7, 1.3), 1)
        random_model = random.randint(1, 5)
        i = np.random.uniform(0, 1)
        if i >= 0.5:
            alpha = random.randint(-5, -1)
        else:
            alpha = random.randint(-5, 5)
        beta = random.randint(-10, 10)
        # print(random_model, i, alpha, beta, gamma, k)

        for trace_group in ink.trace_groups:
            tmp_group_coords = []
            for trace in trace_group.traces:
                tmp_coord = image_processing.geometric_global_transform(trace.coords, random_model, i, alpha, beta,
                                                                        gamma, k)
                tmp_group_coords.append(tmp_coord)
            new_group_coords.append(tmp_group_coords)
        return new_group_coords


if __name__ == '__main__':
    demo_ink = inkml.Ink("../../data/CROHME_full_v2/CROHME2013_data/TrainINKML/HAMEX/formulaire001-equation003.inkml")
    output_dir = 'demo-outputs'
    os.makedirs(output_dir, exist_ok=True)
    for i, (lbl, tgs) in enumerate(InkAugmentor.simple_augment(demo_ink, 5)):
        with open(os.path.join(output_dir, "aug_demo_{:02d}.label.txt".format(i)), "w+") as fout:
            fout.write(lbl)

        plt_draw.plt_clear()
        fig, ax = plt_draw.plt_setup()

        for tg in tgs:
            for t in tg.traces:
                plt_draw.plt_trace_coords(t.coords, ax)

        plt_draw.plt_savefig(os.path.join(output_dir, "aug_demo_{:02d}.png".format(i)))
