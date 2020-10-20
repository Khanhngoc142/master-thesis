import numpy as np
from torch.autograd import Variable

from model.ssdpytorch.ssd import build_ssd
import argparse
import torch
import os
import cv2
import torch.backends.cudnn as cudnn
from utilities.fs import get_source_root
from utilities.data_processing import symbols, symbol2idx
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default=os.path.join(get_source_root(),
                                                            'model/weights/ssd300_14_besteval.pth'),
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default=os.path.join(get_source_root(), 'testing/eval_by_map/aug_geo__/'), type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.5, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--dataset', default='CROHME_2013')
parser.add_argument('--dataset_root', default="data/",
                    help='Dataset root directory path')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder, exist_ok=True)

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


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    label_ids = dets[:, 5]
    sqrt_id = symbol2idx['\\sqrt']
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        if label_ids[i] == sqrt_id:
            inds = np.where(ovr <= 0.65)[0]
        elif label_ids[i] in [symbol2idx['\\sin'], symbol2idx['\\cos'], symbol2idx['\\tan'], symbol2idx['\\log'], symbol2idx['\\lim']]:
            inds = np.where(ovr <= 0.03)[0]
        else:
            inds = np.where(ovr <= thresh)[0]

        if label_ids[i] == symbol2idx['-']:
            inds = np.array([idx for idx in inds if ~(label_ids[idx] == symbol2idx['-'] and ovr[idx] != 0)]).astype(int)
        order = order[inds + 1]

    return keep


def test_net(save_folder, net, cuda, img_paths, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder + 'test.txt'
    num_images = len(img_paths)
    mean = (104, 117, 123)
    for i, img_path in enumerate(img_paths):
        print('Testing {} image {:d}/{:d}....'.format(img_path, i + 1, num_images))
        img = cv2.imread(img_path)
        img = img.astype(np.float32) - mean
        img_pil = Image.open(img_path)

        draw = ImageDraw.Draw(img_pil)
        # to rgb
        img = img[:, :, (2, 1, 0)]

        x = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if cuda:
            x = x.cuda()

        y = net(x)  # forward pass

        # softmax = nn.Softmax(dim=-1)
        # detect = Detect(len(symbols) + 1, 0, 200, 0.01, 0.45)
        #
        # y = (y[0], softmax(y[1]), y[2].type(type(x.data)))
        # y = detect.apply(*y)

        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])
        pred_num = 0

        tmp_img_boxes = []

        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.1:
                # if pred_num == 0:
                #     with open(filename, mode='a') as f:
                #         f.write('PREDICTIONS: ' + '\n')
                score = detections[0, i, j, 0].item()
                label_id = int(i - 1)
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3], score, label_id)

                tmp_img_boxes.append(list(coords))
                j += 1
        if len(tmp_img_boxes) > 0:
            box_ids = nms(np.array(tmp_img_boxes), 0.15)
            boxes = np.array(tmp_img_boxes)[box_ids]
        else:
            boxes = []
        pred_boxes_str = []
        for i, box in enumerate(boxes):
            # if pred_num == 0:
            #     with open(filename, mode='a') as f:
            #         # f.write(img_path.split('/')[-1]+ ' ')
            #         f.write(img_path.replace(get_source_root(), "").lstrip('/').split('.')[0] + '.png' + ' ')
            coords = (box[0], box[1], box[2], box[3])
            score = box[4]
            label_id = int(box[5])
            draw.rectangle(coords, outline='red')
            draw.text((coords[0], coords[1] - 15), symbols[label_id] + " {:.2f}".format(score),
                      fill=(0, 0, 0, 255))
            # img_pil.show()
            pred_num += 1
            pred_boxes_str.append(str(label_id) + ' ' + str(score) + ' ' + ' '.join(str(c) for c in coords))
            # with open(filename, mode='a') as f:
            #     f.write(str(label_id) + ' ' + str(score) + ' ' + ' '.join(str(c) for c in coords) + '\n')
        with open(filename, mode='a') as f:
            # f.write(img_path.split('/')[-1]+ ' ')
            f.write(img_path.replace(get_source_root(), "").lstrip('/').split('.')[0] + '.png' + ' ' + ' '.join(pred_boxes_str) + '\n')

                        # print(pred_num)
        img_pil.save(os.path.join(save_folder, img_path.split('/')[-1].split('.')[0] + '.png'), 'png')


def test():
    # load net
    ssd_net = build_ssd('test', cfg['min_dim'], cfg['num_classes'])
    ssd_net.load_state_dict(torch.load(args.trained_model).state_dict())
    net = ssd_net.float()
    net.eval()

    with torch.no_grad():
        print('Finished loading model!')

        # load data
        test_files = [os.path.join(get_source_root(), "training/data/CROHME_2013_valid/", fname) for fname in
                      os.listdir(os.path.join(get_source_root(), "training/data/CROHME_2013_valid/")) if fname.endswith('.png')]
        if args.cuda:
            net = net.cuda()
            cudnn.benchmark = True
        # evaluation
        test_net(args.save_folder, net, args.cuda, test_files,
                 thresh=args.visual_threshold)


if __name__ == '__main__':
    test()
