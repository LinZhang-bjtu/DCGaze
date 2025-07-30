import random

import torch
import torch.nn as nn
import numpy as np
import math
import copy
import torch.nn.functional as F
from models.CrossAttention import CrossAttention

from torchvision.transforms import Resize

from PIL import Image

import clip
from models.resnet import resnet18
from clip import *

device = "cuda" if torch.cuda.is_available() else "cpu"


class CLIPProcessor:
    def __init__(self, clip=None, preprocess=None):
        self.model = clip
        self.preprocess = preprocess
    def process_image(self, image_array):
        image = image_array
        image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        return image

    def process_texts(self, texts):
        return clip.tokenize(texts).to(device)

    def get_probabilities(self, image, texts):
        image = self.process_image(image)
        text = self.process_texts(texts)

        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.cpu().numpy()

        return probs


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)  # 多头注意力层
        # Implementation of Feedforward model 前馈神经网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):  # 位置编码
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos

    def forward(self, src, pos):
        q = k = self.pos_embed(src, pos)  
        src2 = self.self_attn(q, k, value=src)[0]  
        src = src + self.dropout1(src2)  
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



class Attention(nn.Module):
    def __init__(self,  condition_dim=32*7*7):
        super(Attention, self).__init__()
        self.scale = condition_dim ** -0.5

    def forward(self, x, condition):
        batch_size, channels, height, width = condition.shape
        condition = condition.view(batch_size,   channels, height*height)
        x = x.view(batch_size, channels, height*height)
        w1 = torch.matmul(x, x.transpose(-2, -1))
        w1 = w1 * self.scale
        w1 = F.softmax(w1, dim=-1)
        condition = torch.sigmoid(condition)
        w2 = torch.matmul(condition, condition.transpose(-2, -1))
        w2 = w2 * self.scale
        w2 = F.softmax(w2, dim=-1)
        y = torch.matmul(w1+w2, x)
        y = x+y
        return y


class GazeTR_Model(nn.Module):
    def __init__(self, maps=32, is_AFU=False, adapter=False):
        super(GazeTR_Model, self).__init__()
        nhead = 8
        dim_feature = 7 * 7
        dim_feedforward = 512
        dropout = 0.1
        num_layers = 6
        self.is_AFU = is_AFU
        self.base_model = resnet18(pretrained=False, maps=maps)

        encoder_layer = TransformerEncoderLayer(
            maps,
            nhead,
            dim_feedforward,
            dropout)

        encoder_norm = nn.LayerNorm(maps)

        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))  # 初始化一个cls token

        self.pos_embedding = nn.Embedding(dim_feature + 1, maps)

        self.feed = nn.Linear(maps, 2)  # 线性回归层

        self.loss_op = nn.L1Loss()

        if is_AFU:
            self.clip, self.preprocess = clip.load(clip_v)
            self.clip.float()
            if clip_v == 'RN50':
                self.conv = nn.Sequential(
                    nn.Conv2d(2048, maps, 1),
                    nn.BatchNorm2d(maps),
                    nn.ReLU(inplace=True)
                )
                self.attention = Attention(condition_dim=maps*dim_feature)
            else:
                self.fc = nn.Linear(512, maps)
                self.attention = Attention(condition_dim=maps)
                self.fc2 = nn.Linear(maps*2, maps)

    def forward(self, x_in):
        batch_size = x_in.size(0)
        image_feature, _ = self.base_model(x_in)   
        if self.is_AFU:
            masks = self.clip.encode_image(x_in)
            masks = self.conv(masks)
            image_feature = self.attention(image_feature, masks)

        image_feature = image_feature.flatten(2)  
        image_feature = image_feature.permute(2, 0, 1) 
        cls = self.cls_token.repeat((1, batch_size, 1)) 
        feature = torch.cat([cls, image_feature], 0) 
        position = torch.from_numpy(np.arange(0, 50)).cuda()

        pos_feature = self.pos_embedding(position)  

        feature = self.encoder(feature, pos_feature)  

        feature = feature.permute(1, 2, 0)  

        feature = feature[:, :, 0]  

        gaze = self.feed(feature)  

        return gaze, feature

    def loss(self, x_in, label):
        gaze, feature = self.forward(x_in)
        loss = self.loss_op(gaze, label)
        return loss
