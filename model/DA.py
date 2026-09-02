from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.autograd import Function, Variable

import numpy as np
import cv2


class ImageLabelResizeLayer(nn.Module):
    """
    Resize label to be the same size with the samples
    """

    def __init__(self):
        super(ImageLabelResizeLayer, self).__init__()

    def forward(self, x, need_backprop):
        feats = x.detach().cpu().numpy()
        lbs = need_backprop.detach().cpu().numpy()
        gt_blob = np.zeros((lbs.shape[0], feats.shape[2], feats.shape[3], 1), dtype=np.float32)  # 特征图层面打标签
        for i in range(lbs.shape[0]):
            lb = np.array([lbs[i]])
            lbs_resize = cv2.resize(lb, (feats.shape[3], feats.shape[2]), interpolation=cv2.INTER_NEAREST)
            gt_blob[i, 0:lbs_resize.shape[0], 0:lbs_resize.shape[1], 0] = lbs_resize

        channel_swap = (0, 3, 1, 2)
        gt_blob = gt_blob.transpose(channel_swap)
        y = Variable(torch.from_numpy(gt_blob)).cuda()
        return y


class InstanceLabelResizeLayer(nn.Module):
    def __init__(self):
        super(InstanceLabelResizeLayer, self).__init__()
        self.minibatch = 256

    def forward(self, x, need_backprop):
        # lbs.size() --> ([1])
        feats = x.data.cpu().numpy()
        lbs = need_backprop.data.cpu().numpy()

        resized_lbs = np.ones((feats.shape[0], 1), dtype=np.float32)
        # lbs.shape[0] -> 1
        for i in range(lbs.shape[0]):
            resized_lbs[i * self.minibatch:(i + 1) * self.minibatch] = lbs[i]

        y = torch.from_numpy(resized_lbs).cuda()

        return y


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha = 0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.neg() * ctx.alpha
        return output


def grad_reverse(x):
    return GRLayer.apply(x)


class _ImageDA_res(nn.Module):
    def __init__(self, dim):
        super(_ImageDA_res, self).__init__()
        self.dim = dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1, bias=False)
        self.Conv2 = nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=False)
        self.reLu = nn.ReLU(inplace=False)
        self.LabelResizeLayer = ImageLabelResizeLayer()
        self.refineconv = nn.Sequential(Bottleneck(self.dim, self.dim // 4),
                                        Bottleneck(self.dim, self.dim // 4))

    def forward(self, x, need_backprop):
        x = grad_reverse(x)
        x = self.refineconv(x)
        x = self.reLu(self.Conv1(x))
        x = self.Conv2(x)
        label = self.LabelResizeLayer(x, need_backprop)
        return x, label


class _ImageDA(nn.Module):
    def __init__(self, dim):
        super(_ImageDA, self).__init__()
        self.dim = dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1, bias=False)
        # self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.Conv2 = nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=False)
        self.reLu = nn.ReLU(inplace=False)
        self.LabelResizeLayer = ImageLabelResizeLayer()

    def forward(self, x, need_backprop):
        x = grad_reverse(x)
        x = self.reLu(self.Conv1(x))
        x = self.Conv2(x)
        label = self.LabelResizeLayer(x, need_backprop)

        return x, label


class _ImageDA_noGRL(nn.Module):
    def __init__(self, dim):
        super(_ImageDA_noGRL, self).__init__()
        self.dim = dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1, bias=False)
        # self.Conv2=nn.Conv2d(512,2,kernel_size=1,stride=1,bias=False)
        self.Conv2 = nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=False)
        self.reLu = nn.ReLU(inplace=False)
        self.LabelResizeLayer = ImageLabelResizeLayer()

    def forward(self, x, need_backprop):
        x = self.reLu(self.Conv1(x))
        x = self.Conv2(x)
        label = self.LabelResizeLayer(x, need_backprop)

        return x, label


class _InstanceDA(nn.Module):
    def __init__(self, dim):
        super(_InstanceDA, self).__init__()
        self.dc_ip1 = nn.Linear(dim, dim//2)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(dim//2, dim//4)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer = nn.Linear(dim//4, 1)
        self.LabelResizeLayer = InstanceLabelResizeLayer()

        # self.dc_ip1 = nn.Linear(dim, dim)
        # self.dc_relu1 = nn.ReLU()
        # self.dc_drop1 = nn.Dropout(p=0.5)
        # self.cam_classifer = nn.Linear(dim, cam_cls)
        # self.id_classifer = nn.Linear(dim, num_cls)

    def forward(self, x, need_backprop):
        x = grad_reverse(x)
        x = self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x = self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        x = F.sigmoid(self.classifer(x))

        label = self.LabelResizeLayer(x, need_backprop)
        return x, label


class _InstanceDA_En(nn.Module):
    def __init__(self, dim):
        super(_InstanceDA_En, self).__init__()
        self.dc_ip1 = nn.Linear(dim, dim//2)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(dim//2, dim//4)
        self.dc_relu2 = nn.ReLU()
        self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer = nn.Linear(dim//4, 1)
        self.LabelResizeLayer = InstanceLabelResizeLayer()

        # self.dc_ip1 = nn.Linear(dim, dim)
        # self.dc_relu1 = nn.ReLU()
        # self.dc_drop1 = nn.Dropout(p=0.5)
        # self.cam_classifer = nn.Linear(dim, cam_cls)
        # self.id_classifer = nn.Linear(dim, num_cls)

    def forward(self, x, need_backprop):
        x = grad_reverse(x)
        x = self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x = self.dc_drop2(self.dc_relu2(self.dc_ip2(x)))
        # cam_logit = self.cam_classifer(x)
        # id_logit = self.id_classifer(x)
        x = F.sigmoid(self.classifer(x))

        label = self.LabelResizeLayer(x, need_backprop)
        return x, label
