#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    test.py
# @Project:     GuideNet
# @Author:      jie
# @Time:        2021/3/16 4:47 PM

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import yaml
from easydict import EasyDict as edict
import datasets
import encoding

def test():
    net.eval()
    for batch_idx, (rgb, lidar, _, idx, ori_size) in enumerate(testloader):
        with torch.no_grad():
            if config.tta:
                rgbf = torch.flip(rgb, [-1])
                lidarf = torch.flip(lidar, [-1])
                rgbs = torch.cat([rgb, rgbf], 0)
                lidars = torch.cat([lidar, lidarf], 0)
                rgbs, lidars = rgbs.cuda(), lidars.cuda()
                depth_preds, = net(rgbs, lidars)
                depth_pred, depth_predf = depth_preds.split(depth_preds.shape[0] // 2)
                depth_predf = torch.flip(depth_predf, [-1])
                depth_pred = (depth_pred + depth_predf) / 2.
            else:
                rgb, lidar = rgb.cuda(), lidar.cuda()
                depth_pred, = net(rgb, lidar)
            depth_pred[depth_pred < 0] = 0
        depth_pred = depth_pred.cpu().squeeze(1).numpy()
        idx = idx.cpu().squeeze(1).numpy()
        ori_size = ori_size.cpu().numpy()
        name = [testset.names[i] for i in idx]
        save_result(config, depth_pred, name, ori_size)


if __name__ == '__main__':
    # config_name = 'GN.yaml'
    config_name = 'GNS.yaml'
    with open(os.path.join('configs', config_name), 'r') as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)
    config = edict(config_data)
    from utils import *

    transform = init_aug(config.test_aug_configs)
    key, params = config.data_config.popitem()
    dataset = getattr(datasets, key)
    testset = dataset(**params, mode='test', transform=transform, return_idx=True, return_size=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, num_workers=config.num_workers,
                                             shuffle=False, pin_memory=True)
    print('num_test = {}'.format(len(testset)))
    net = init_net(config)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    net.cuda()
    net = encoding.parallel.DataParallelModel(net)
    net = resume_state(config, net)
    test()
