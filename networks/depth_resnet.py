
from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from collections import OrderedDict
from layers import *
from timm.models.layers import trunc_normal_

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class DepthEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(DepthEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

# class DepthEncoder(nn.Module):
#     """Pytorch module for a resnet encoder
#     """
#     def __init__(self, num_layers, pretrained):
#         super(DepthEncoder, self).__init__()

#         self.num_ch_enc = np.array([64, 64, 128, 256, 512])

#         resnets = {18: models.resnet18,
#                    34: models.resnet34,
#                    50: models.resnet50,
#                    101: models.resnet101,
#                    152: models.resnet152}

#         if num_layers not in resnets:
#             raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

#         self.encoder = resnets[num_layers](pretrained)

#         if num_layers > 34:
#             self.num_ch_enc[1:] *= 4
        
#         self.conv1 = nn.Conv2d(3, self.num_ch_enc[0], kernel_size=7, stride=1, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.num_ch_enc[0])

#         self.conv1.weight = nn.Parameter(self.encoder.conv1.weight.clone())
#         self.bn1.weight = nn.Parameter(self.encoder.bn1.weight.clone())
#         self.bn1.bias = nn.Parameter(self.encoder.bn1.bias.clone())


#     def forward(self, input_image):

#         self.features = []
#         x = (input_image - 0.45) / 0.225
#         context = self.encoder.relu(self.bn1(self.conv1(x)))
#         x = self.encoder.conv1(x)
#         x = self.encoder.bn1(x)
#         self.features.append(self.encoder.relu(x))
#         self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))

#         self.features.append(self.encoder.layer2(self.features[-1]))
#         self.features.append(self.encoder.layer3(self.features[-1]))
#         self.features.append(self.encoder.layer4(self.features[-1]))

#         self.features = [context] + self.features
#         return self.features
class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

# class DepthDecoder(nn.Module):
#     def __init__(self, num_ch_enc, scales=range(4), columns=3,num_output_channels=1):
#         super(DepthDecoder, self).__init__()

#         self.num_output_channels = num_output_channels
#         self.scales = scales
#         self.num_ch_enc = list(num_ch_enc)
#         self.num_ch_enc = [self.num_ch_enc[0]] + self.num_ch_enc
#         self.grid_ch = [ch//2 for ch in self.num_ch_enc]
#         self.grid_ch[0] = 16
#         self.columns = columns

#         self.convs = OrderedDict()
#         for i in range(len(self.grid_ch)):
#             for j in range(self.columns):
#                 if j == 0:
#                     self.convs[('lateral', i, j)] = ResidualBlock(self.num_ch_enc[i], self.grid_ch[i], stride=1)
#                 else:
#                     self.convs[('lateral', i, j)] = ResidualBlock(self.grid_ch[i], self.grid_ch[i], stride=1)

#                 if i < len(self.grid_ch)-1:
#                     self.convs[('upconv', i, j)] = UpsampleBlock(self.grid_ch[i+1], self.grid_ch[i], stride=1)

#             if i in self.scales:
#                 self.convs[("dispconv", i)] = Conv3x3(self.grid_ch[i], self.num_output_channels)

#         self.decoder = nn.ModuleList(list(self.convs.values()))
#         self.sigmoid = nn.Sigmoid()

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             trunc_normal_(m.weight, std=.02)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, input_features):
#         self.outputs = {}
#         lateral_tmp = [0 for _ in range(self.columns)]
      
#         for i in range(len(self.grid_ch)-1, -1, -1):
#             x = input_features[i]
#             for j in range(self.columns):
#                 x = self.convs[('lateral', i, j)](x)
#                 if i < len(self.grid_ch)-1:
#                     x = x + lateral_tmp[j]
#                 if i > 0:
#                     lateral_tmp[j] = self.convs[('upconv', i-1, j)](x)

#             if i in self.scales:
#                 self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
  
#         return self.outputs