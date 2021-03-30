#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    train.py
# @Project:     GuideNet
# @Author:      jie
# @Time:        2021/3/14 7:50 PM

import os
import torch
import yaml
from easydict import EasyDict as edict


def train(epoch):
    global iters
    Avg = AverageMeter()
    for batch_idx, (rgb, lidar, depth) in enumerate(trainloader):
        if epoch >= config.test_epoch and iters % config.test_iters == 0:
            test()
        net.train()
        rgb, lidar, depth = rgb.cuda(), lidar.cuda(), depth.cuda()
        optimizer.zero_grad()
        output = net(rgb, lidar)
        loss = criterion(output, depth).mean()
        loss.backward()
        optimizer.step()
        Avg.update(loss.item())
        iters += 1
        if config.vis and batch_idx % config.vis_iters == 0:
            print('Epoch {} Idx {} Loss {:.4f}'.format(epoch, batch_idx, Avg.avg))


def test():
    global best_metric
    Avg = AverageMeter()
    net.eval()
    for batch_idx, (rgb, lidar, depth) in enumerate(testloader):
        rgb, lidar, depth = rgb.cuda(), lidar.cuda(), depth.cuda()
        with torch.no_grad():
            output = net(rgb, lidar)
            prec = metric(output, depth).mean()
        Avg.update(prec.item(), rgb.size(0))
    if Avg.avg < best_metric:
        best_metric = Avg.avg
        save_state(config, net)
        print('Best Result: {:.4f}\n'.format(best_metric))


if __name__ == '__main__':
    # config_name = 'GN.yaml'
    config_name = 'GNS.yaml'
    with open(os.path.join('configs', config_name), 'r') as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)
    config = edict(config_data)
    print(config.name)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_id) for gpu_id in config.gpu_ids])
    from utils import *

    init_seed(config)
    trainloader, testloader = init_dataset(config)
    net = init_net(config)
    criterion = init_loss(config)
    metric = init_metric(config)
    net, criterion, metric = init_cuda(net, criterion, metric)
    optimizer = init_optim(config, net)
    lr_scheduler = init_lr_scheduler(config, optimizer)
    iters = 0
    best_metric = 100
    for epoch in range(config.start_epoch, config.nepoch):
        train(epoch)
        lr_scheduler.step()
    print('Best Results: {:.4f}\n'.format(best_metric))
