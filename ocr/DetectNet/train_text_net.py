from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import pickle
from functools import partial


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

from args import argument_parser, image_dataset_kwargs, optimizer_kwargs
from torchtext.data_manager import DetectImageManager
from torchtext import models
from torchtext.losses import TextLoss
from torchtext.utils.iotools import save_checkpoint, check_isfile
from torchtext.utils.avgmeter import AverageMeter
from torchtext.utils.loggers import Logger, RankLogger
from torchtext.utils.torchtools import count_num_param, open_all_layers, open_specified_layers
from torchtext.utils.generaltools import set_random_seed
from torchtext.optimizers import init_optimizer


# global variables
parser = argument_parser()
args = parser.parse_args()


def main():
    global args
    glob_min_loss = 0.2
    args.source_names = ['total-text']
    # args.source_names = ['mars']
    args.target_names = ['total-text']
    args.height = 512
    args.width = 512
    args.optim = 'amsgrad'
    # args.label_smooth = True
    args.lr = 0.0003
    # args.weight_decay = 1e-4
    # args.gamma = 0.1
    args.max_epoch = 100
    args.stepsize = [20,60]
    args.train_batch_size = 1
    # args.workers=1
    args.arch = 'resnet50'
    args.save_dir = 'log/resnet50-textfield'
    args.gpu_devices = '0'
    # args.lambda_xent = 1
    # args.lambda_htri = 1
    # args.resume = 'log/resnet50-textfield/quick_save_checkpoint_ep3.pth.tar'
    # args.load_weights = 'log/resnet50-textfield/quick_save_checkpoint_ep21.pth.tar'
    # args.start_epoch = 47

    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    log_name = 'log_test.txt' if args.evaluate else 'log_train.txt'
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        print("Currently using CPU, however, GPU is highly recommended")

    print("Initializing image data manager")
    dm = DetectImageManager(use_gpu, **image_dataset_kwargs(args))
    trainloader = dm.return_dataloaders()

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch)
    print("Model size: {:.3f} M".format(count_num_param(model)))

    criterion = TextLoss()

    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=args.stepsize, gamma=args.gamma)

    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    if args.load_weights and check_isfile(args.load_weights):
        # load pretrained weights but ignore layers that don't match in size
        checkpoint = torch.load(args.load_weights, pickle_module=pickle)
        pretrain_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items(
        ) if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        print("Loaded pretrained weights from '{}'".format(args.load_weights))

    if args.resume and check_isfile(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch'] + 1
        print("Loaded checkpoint from '{}'".format(args.resume))
        print("- start_epoch: {}\n- rank1: {}".format(args.start_epoch,
                                                      checkpoint['rank1']))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    start_time = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    train_time = 0
    print("=> Start training")

    for epoch in range(0, args.start_epoch):
        scheduler.step()
        continue

    if args.fixbase_epoch > 0:
        print("Train {} for {} epochs while keeping other layers frozen".format(
            args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            start_train_time = time.time()
            train(epoch, model, criterion, optimizer,
                  trainloader, use_gpu, fixbase=True)
            train_time += round(time.time() - start_train_time)

        print("Done. All layers are open to train for {} epochs".format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)

    for epoch in range(args.start_epoch, args.max_epoch):
        start_train_time = time.time()
        local_loss = train(epoch, model, criterion,
                           optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)

        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("=> Test")

        if epoch % 10 == 0 or local_loss < glob_min_loss:
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            if local_loss < glob_min_loss:
                glob_min_loss = local_loss
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': 0,
                'epoch': epoch,
                'avg_loss': local_loss,
            }, False, osp.join(args.save_dir, 'quick_save_checkpoint_ep' + str(epoch + 1) + '.pth.tar'))
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(
        elapsed, train_time))
    ranklogger.show_summary()


def train(epoch, model, criterion, optimizer, trainloader, use_gpu, fixbase=False):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    if fixbase or args.always_fixbase:
        open_specified_layers(model, args.open_layers)
    else:
        open_all_layers(model)

    end = time.time()
    for batch_idx, (imgs, vecs, weights) in enumerate(trainloader):
        data_time.update(time.time() - end)
        if use_gpu:
            imgs, vecs, weights = imgs.cuda(), vecs.cuda(), weights.cuda()

        outputs = model(imgs)

        loss = criterion(outputs, vecs, weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(loss.item(), imgs.size(0))
        # del loss, imgs, vecs, weights
        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                      epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                      data_time=data_time, loss=losses))

        end = time.time()
    return losses.avg


if __name__ == '__main__':
    main()
