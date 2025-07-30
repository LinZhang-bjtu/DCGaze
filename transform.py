import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2

def Gaze360_transform():
    transform = torchvision.transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def EYEDIAP_transform():
    transform = torchvision.transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform

def mpiifacegaze_transform():
    scale = torchvision.transforms.Lambda(lambda x: x.astype(np.float32) / 255)
    identity = torchvision.transforms.Lambda(lambda x: x)
    size = 224
    if size != 448:
        resize = torchvision.transforms.Lambda(
            lambda x: cv2.resize(x, (size, size)))
    else:
        resize = identity
    to_gray = identity
    transform = torchvision.transforms.Compose([
        resize,
        to_gray,
        torchvision.transforms.Lambda(lambda x: x.transpose(2, 0, 1)),
        scale,
        torch.from_numpy,
        torchvision.transforms.Normalize(mean=[0.406, 0.456, 0.485],
                                         std=[0.225, 0.224, 0.229]),
    ])
    return transform