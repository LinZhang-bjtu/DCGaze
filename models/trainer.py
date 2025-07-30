import os
import random
import sys
from collections import defaultdict
from itertools import combinations

import torch
import torch.nn as nn
import numpy as np
import math
import copy
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

from torch.autograd import Variable


import clip
from models.resnet import resnet18
from clip import *
from models.CrossAttention import CrossAttention
import torch.backends.cudnn as cudnn
from models.GazeTR_model import GazeTR_Model
from models.AlignLoss import AlignLoss

device = "cuda" if torch.cuda.is_available() else "cpu"


def Mask(nb_batch):
    bar = []
    drop = 5
    for i in range(1):
        foo = [1] * (32-drop) + [0] * drop
        random.shuffle(foo)  #### generate mask
        bar += foo
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch, 32, 1, 1)
    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    bar = Variable(bar)
    return bar


class my_MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(my_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        input = input.transpose(3, 1)
        input = F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
        input = input.transpose(3, 1).contiguous()

        return input

    def __repr__(self):
        kh, kw = _pair(self.kernel_size)
        dh, dw = _pair(self.stride)
        padh, padw = _pair(self.padding)
        dilh, dilw = _pair(self.dilation)
        padding_str = ', padding=(' + str(padh) + ', ' + str(padw) + ')' \
            if padh != 0 or padw != 0 else ''
        dilation_str = (', dilation=(' + str(dilh) + ', ' + str(dilw) + ')'
                        if dilh != 0 and dilw != 0 else '')
        ceil_str = ', ceil_mode=' + str(self.ceil_mode)
        return self.__class__.__name__ + '(' \
            + 'kernel_size=(' + str(kh) + ', ' + str(kw) + ')' \
            + ', stride=(' + str(dh) + ', ' + str(dw) + ')' \
            + padding_str + dilation_str + ceil_str + ')'


class Trainer(nn.Module):
    def __init__(self, train_config):
        super(Trainer, self).__init__()
        maps = 32
        is_AFU = train_config.is_AFU
        self.a = train_config.a
        self.b = train_config.b

        self.base_model = GazeTR_Model(maps=maps, is_AFU=is_AFU)

        pretrain = train_config.pretrain
        if pretrain.enable and pretrain.device:
            checkpoint = torch.load(
                pretrain.path,
                map_location={f"cuda:0": f"cuda:{train_config.device}"}
            )
            new_pth = self.base_model.state_dict()  
            pretrained_dict = {}  
            for k, v in checkpoint.items():
                for kk in new_pth.keys():
                    if kk in k:
                        pretrained_dict[kk] = v
                        break
            new_pth.update(pretrained_dict)
            self.base_model.load_state_dict(new_pth)
        elif pretrain.enable and not pretrain.device:
            checkpoint = torch.load(pretrain.path)
            new_pth = self.base_model.state_dict()  
            pretrained_dict = {}  
            for k, v in checkpoint.items():
                for kk in new_pth.keys():
                    if kk in k:
                        pretrained_dict[kk] = v
                        break
            new_pth.update(pretrained_dict)
            self.base_model.load_state_dict(new_pth)

        self.loss_op = nn.L1Loss()
        self.AlignLoss = AlignLoss(maps=maps,learning_prompt=learn_prompt)
        self.grade = train_config.grade


    def forward(self, x_in):
        gaze, self.feature = self.base_model(x_in)

        return gaze

    def loss(self, x_in, label):

        gaze = self.forward(x_in)
        loss1 = self.loss_op(gaze, label)

        loss_align = self.AlignLoss(self.feature, label, grade = self.grade)


        feature = self.feature
        mask = Mask(feature.size(0))
        branch_1 = feature.reshape(feature.size(0), feature.size(1), 1, 1) * mask
        branch_1 = my_MaxPool2d(kernel_size=(1, 16), stride=(1, 16))(branch_1)
        branch_1 = branch_1.view(branch_1.size(0), -1)
        loss_mask = self.loss_op(branch_1, label)

        loss = loss1 + self.a * loss_mask + self.b * loss_align

        return gaze,loss
