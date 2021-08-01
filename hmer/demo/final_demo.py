# load image
# segment using ssd
# structural analysis using draculae

import os
import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from model.ssdpytorch.ssd import build_ssd
from utilities.fs import get_source_root
from utilities.data_processing import symbols, symbol2idx
from utilities.image_processing import IMG_NORM_MEAN
from testing.test_ssd import nms
from utilities import BB_To_Tree


def load_image(path):
    """
    Load image and scale to correct size (300x300)
    Args:
        path (:obj:`str`): path of the input image
    Returns:
        :obj:`numpy.ndarray`: Input to feed into SSD
    """
    img = cv2.imread(path)  # use opencv to read image
    img = img[:, :, (2, 1, 0)]  # transpose to RGB due to cv2.imread return image in form of BGR

    # Scale image
    img_h, img_w = img.shape[:2]
    if img_w < 300 and img_h < 300:
        if img_w > img_h:
            img = ResizeWithAspectRatio(img, width=300)
        else:
            img = ResizeWithAspectRatio(img, height=300)

    # Pad image to 300x300
    img_h, img_w = img.shape[:2]
    if img_w < 300 or img_h < 300:
        new_img = np.full((300, 300, 3), 255, dtype=np.uint8)
        center_x = (300 - img_w) // 2
        center_y = (300 - img_h) // 2
        new_img[center_y:center_y+img_h, center_x:center_x+img_w] = img
        img = new_img

    return img


def prepare_img(img):
    """
    Prepare image to feed into NN
    Args:
        img (:obj:`numpy.ndarray`): input image

    Returns:
        :obj:`torch.Tensor`: tensor to feed into NN
    """
    img = img - IMG_NORM_MEAN  # normalize image
    img = torch.from_numpy(img.astype(np.float32))  # load into pytorch datatype
    img = img.permute(2, 0, 1)  # permute into CxHxW to feed into NN
    img = img.unsqueeze(0)  # turn into a 1-sample batch
    return img


def display_img(img):
    img = Image.fromarray(img, "RGB")
    img.show()


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize image to specific size while maintaining aspect ratio.
    Prioritize desired width over desired height.
    Source: https://stackoverflow.com/a/58126805/6262115
    Args:
        image (:obj:`numpy.ndarray`): Input image of shape (H,W[,C])
        width (int, optional): Desired width
        height (int, optional): Desired height
        inter (:obj:`str`, optional): Interpolation style

    Returns:
        :obj:`numpy.ndarray`: Resized image

    """
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def latex_generator(line, is_pred=True):
    generator = BB_To_Tree.BBParser()
    return generator.process(line, is_pred)


# noinspection SpellCheckingInspection
def main(model, image, cuda, show_original_image):
    cfg = {
        'num_classes': len(symbols) + 1,
        'lr_steps': (280000, 360000, 400000),
        'feature_maps': [38, 19, 10, 5, 3, 1],
        'min_dim': 300,
        'steps': [8, 16, 32, 64, 100, 300],
        'min_sizes': [21, 45, 99, 153, 207, 261],
        'max_sizes': [45, 99, 153, 207, 261, 315],
        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'CROHME',
    }

    cuda = args.cuda and torch.cuda.is_available()

    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Load model
    print("Load model.")
    ssd_net = build_ssd('test', cfg['min_dim'], cfg['num_classes'])
    ssd_net.load_state_dict(torch.load(model, map_location=torch.device('cpu') if not cuda else None).state_dict())

    net = ssd_net.float()
    net.eval()

    # Prepare input
    print("Prepare input.")
    img = load_image(args.image)
    if show_original_image:
        display_img(img)
    x = prepare_img(img)
    if cuda:
        x = x.cuda()

    # Get predictions
    print("Get predictions.")
    with torch.no_grad():
        y = net(x)  # forward pass

    detections = y.data

    scale = torch.Tensor([300, 300, 300, 300])
    tmp_img_boxes = []
    for i in range(detections.size(1)):
        j = 0
        while True:
            score = detections[0, i, j, 0].item()
            if score < 0.1:
                break
            label_id = int(i - 1)
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            coords = [*[p for p in pt[:4]], score, label_id]

            tmp_img_boxes.append(coords)
            j += 1

    tmp_img_boxes = np.array(tmp_img_boxes)

    if len(tmp_img_boxes) > 0:
        box_ids = nms(tmp_img_boxes, 0.15)
        boxes = tmp_img_boxes[box_ids]
    else:
        boxes = []

    # Visualize prediction
    pred_boxes_str = []
    img_pil = Image.fromarray(img, "RGB")
    draw = ImageDraw.Draw(img_pil)
    for box in boxes:
        coords = box[:4].tolist()
        score = box[4]
        label_id = int(box[5])
        draw.rectangle(coords, outline='red')
        draw.text((coords[0], coords[1] - 15),
                  symbols[label_id] + " {:.2f}".format(score),
                  fill=(0, 0, 0, 255))
        pred_boxes_str.append(str(label_id) + ' ' + str(score) + ' ' + ' '.join(str(c) for c in coords))
    img_pil.show()

    # parse into string
    pred_str = image + ' ' + ' '.join(pred_boxes_str)
    return latex_generator(pred_str, is_pred=True)


def build_args():
    parser = argparse.ArgumentParser(description="Final Demo")
    parser.add_argument("model", type=str, help='SSD model used')
    parser.add_argument("image", type=str, help='Image\'s path')
    parser.add_argument("--cuda", default=False, help='Use cuda', action='store_true')
    parser.add_argument("--show-original-image", default=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cmd_args = build_args()
    main(cmd_args.model, cmd_args.image, cmd_args.cuda, cmd_args.show_original_image)
