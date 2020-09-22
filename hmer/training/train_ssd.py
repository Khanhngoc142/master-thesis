from model.ssdpytorch.utils.augmentations import SSDAugmentation
from model.ssdpytorch.layers.modules import MultiBoxLoss
from model.ssdpytorch.ssd import build_ssd
from model.ssdpytorch.data import *
import os
import sys
import datetime as dt
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse

from utilities.fs import get_source_root, get_path
from extractor.crohme_parser.dataset import CROHMEDetection4SSD
from utilities.data_processing import symbols


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='CROHME_2013')
parser.add_argument('--dataset_root', default=os.path.join(get_source_root(), "demo-outputs/data/CROHME_2013_train"),
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--epochs', default=10, type=int,
                    help='Number of epoch(s)')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='Resume training at this epoch')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
# parser.add_argument('--visdom', default=False, type=str2bool,
#                     help='Use visdom for loss visualization')
# parser.add_argument('--visdom_name', type=str, default=None)
parser.add_argument('--save_folder', default=os.path.join(get_source_root(), 'model/weights/'),
                    help='Directory for saving checkpoint models')
parser.add_argument('--tensorboard_logdir', type=str, default=None)
parser.add_argument('--run_id', type=str, default=dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
parser.add_argument('--save_epoch', default=3, type=int,
                    help='Save model per n epoch(s)')
parser.add_argument('--validset_root', type=str, default=None)
parser.add_argument('--eval_epoch', type=int, default=3)
parser.add_argument('--verbose', type=int, default=10)
parser.add_argument('--model_name', type=str, default='ssd300')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.tensorboard_logdir is not None:
    import tensorflow as tf


def get_num_batch(dataset, batch_size):
    return len(dataset) // batch_size + (0 if len(dataset) % batch_size == 0 else 1)


def train():
    # load dataset
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

    # Prepare data
    print('Loading the dataset...')
    dataset = CROHMEDetection4SSD(root=args.dataset_root)
    if args.validset_root is not None:
        validset = CROHMEDetection4SSD(root=args.validset_root)
    else:
        validset = None
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    if validset is not None:
        valid_data_loader = data.DataLoader(validset, args.batch_size,
                                            num_workers=args.num_workers,
                                            shuffle=True, collate_fn=detection_collate,
                                            pin_memory=True)
    else:
        valid_data_loader = None

    file_writers = {}
    if args.tensorboard_logdir is not None:
        log_dir = get_path(os.path.join(args.tensorboard_logdir, args.model_name + '-' + args.run_id))

        for k in ['loss', 'loc_loss', 'conf_loss']:
            file_writers[f'train_{k}'] = tf.summary.create_file_writer(os.path.join(log_dir, f"train_{k}"))
            if validset is not None:
                file_writers[f'valid_{k}'] = tf.summary.create_file_writer(os.path.join(log_dir, f"valid_{k}"))

    # build network
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net.float()

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    num_batch = get_num_batch(dataset, args.batch_size)
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    print("Start training")
    epoch = None
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        # trigger train mode
        net.train()

        # create batch iterator
        batch_iterator = iter(data_loader)

        # reset epoch loss counters
        loc_loss = 0
        conf_loss = 0
        epoch_time = 0
        # for iteration, (images, targets) in enumerate(batch_iterator):
        for iteration in range(num_batch):
            images, targets = next(batch_iterator)

            if epoch * num_batch + iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            if args.cuda:
                images = Variable(images.cuda())
                with torch.no_grad():
                    targets = [Variable(ann.cuda()) for ann in targets]
            else:
                images = Variable(images)
                with torch.no_grad():
                    targets = [Variable(ann) for ann in targets]

            t0 = time.time()
            # forward
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            t1 = time.time()
            epoch_time += (t1 - t0)

            if args.tensorboard_logdir is not None:
                tb_update_loss(file_writers, epoch * num_batch + iteration, loss_l.item(), loss_c.item(), 'train', 'batch_loss')

        # average loss
        loc_loss /= num_batch
        conf_loss /= num_batch

        # update epoch_plot
        if args.tensorboard_logdir is not None:
            tb_update_loss(file_writers, epoch, loc_loss, conf_loss, 'train', 'epoch_loss')

        if epoch % args.verbose == 0:
            print(f'TRAIN @ epoch {epoch}' + ' || Loss: %.4f ||' % (loc_loss + conf_loss) + f' time: {epoch_time:.3f} sec.')

        if epoch == 0 or epoch % args.save_epoch == 0:
            print('Saving state, epoch:', epoch)
            save_model(ssd_net, epoch, args.save_folder, args.run_id, model_name=args.model_name)

        # EVAL
        if valid_data_loader is not None and ((epoch % args.eval_epoch == 0) or (epoch == (args.start_epoch + args.epochs - 1))):
            with torch.no_grad():
                net.eval()  # trigger eval mode
                valid_iterator = iter(valid_data_loader)
                valid_loc_loss = 0
                valid_conf_loss = 0
                valid_num_batch = get_num_batch(validset, args.batch_size)
                for _ in range(valid_num_batch):
                    images, targets = next(valid_iterator)

                    if args.cuda:
                        images = Variable(images.cuda())
                        with torch.no_grad():
                            targets = [Variable(ann.cuda()) for ann in targets]
                    else:
                        images = Variable(images)
                        with torch.no_grad():
                            targets = [Variable(ann) for ann in targets]
                    # forward
                    out = net(images)
                    # backprop
                    loss_l, loss_c = criterion(out, targets)
                    valid_loc_loss += loss_l.item()
                    valid_conf_loss += loss_c.item()
                valid_loc_loss /= valid_num_batch
                valid_conf_loss /= valid_num_batch

                print(f'EVAL @ epoch {epoch}' + ' || Loss: %.4f ||' % (valid_loc_loss + valid_conf_loss))

                if args.tensorboard_logdir is not None:
                    tb_update_loss(file_writers, epoch, valid_loc_loss, valid_conf_loss, 'valid', 'epoch_loss')

    save_model(ssd_net, epoch if epoch is not None else 0, args.save_folder, args.run_id, model_name=args.model_name)


def save_model(net, it, save_folder, run, model_name='ssd300'):
    save_folder = get_path(save_folder.rstrip('/') + '-' + run)

    save_path = os.path.join(save_folder, model_name + '_' + str(it) + '.pth')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(net, save_path)


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def tb_update_loss(file_writers, it, loc, conf, phase, metric):
    with file_writers[f'{phase}_loc_loss'].as_default():
        tf.summary.scalar(name=metric, data=loc, step=it)
    with file_writers[f'{phase}_conf_loss'].as_default():
        tf.summary.scalar(name=metric, data=conf, step=it)
    with file_writers[f'{phase}_loss'].as_default():
        tf.summary.scalar(name=metric, data=loc + conf, step=it)


if __name__ == '__main__':
    train()
