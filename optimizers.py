#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    optimizers.py
# @Project:     GuideNet
# @Author:      jie
# @Time:        2021/3/15 4:59 PM
"""
This is a fixup as pytorch 1.4.0 can not import AdamW directly from torch.optim
"""

from torch.optim import *
from torch.optim.adamw import AdamW