import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageTk

from model.ssdpytorch.ssd import build_ssd
from testing.test_ssd import nms
from utilities import BB_To_Tree
from utilities.data_processing import symbols
from utilities.image_processing import IMG_NORM_MEAN
import PySimpleGUI as sg


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
def get_result(model, image, cuda):
    cuda = cuda and torch.cuda.is_available()

    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    net = model

    # Prepare input
    print("Prepare input.")
    img = load_image(image)
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
    # img_pil.show()

    # parse into string
    pred_str = image + ' ' + ' '.join(pred_boxes_str)
    return img_pil, latex_generator(pred_str, is_pred=True)


def load_model(model_path, cuda):
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

    print("Load model.")
    ssd_net = build_ssd('test', cfg['min_dim'], cfg['num_classes'])
    ssd_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu') if not cuda else None).state_dict())

    net = ssd_net.float()
    net.eval()

    return net


def main():
    layout = [
        [
            sg.Text("Image File"),
            sg.In(size=(25, 1), enable_events=True, key="-FILE-"),
            sg.FileBrowse(),
        ],
        [
            sg.Column([
                [sg.Text("Original Image", size=(40, 1))],
                [sg.Text(size=(40, 1), key="-TOUT1-")],
                [sg.Image(key="-IMAGE1-", size=(300, 300))],
            ], vertical_alignment="top"),
            sg.VSeperator(),
            sg.Column([
                [sg.Text("Result:", size=(40, 1))],
                [sg.Text(size=(40, 1), key="-TOUT2-")],
                [sg.Image(key="-IMAGE2-", size=(300, 300))],
            ], vertical_alignment="top"),
        ]
    ]

    ssd_model = load_model(
        "/home/hoangnqk/Workplace/git/master-thesis/hmer/model/weights/model_datav1/CROHME_2013_aug_extra2_data-20201019-221615/ssd300_24_best_eval.pth",
        cuda=False)
    window = sg.Window("Final DEMO", layout)

    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        if event == "-FILE-":
            try:
                filename = values["-FILE-"]
                window["-TOUT1-"].update(os.path.basename(filename))
                window["-IMAGE1-"].update(filename=filename)
                seg_img, ltx_str = get_result(ssd_model, filename, cuda=False)
                window["-IMAGE2-"].update(data=ImageTk.PhotoImage(seg_img))
                window["-TOUT2-"].update(ltx_str)
            except:
                pass

    window.close()


if __name__ == "__main__":
    main()
