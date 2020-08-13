# -*- coding: utf-8 -*-
'''
*******************************************************************************
*                                   GDDI                                      *
*******************************************************************************
* File: mbn_v1.py
* Author: rzyang
* Date: 2020/08/05
* Description: MobileNetV1 backbone, refer to https://arxiv.org/pdf/1704.04861.
*******************************************************************************
'''
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES


class MbnV1Module(nn.Module):
    '''
    MobileNetV1 module:
    ___________________
    |   3x3 dw conv   |
    |       BN        |
    |      ReLU       |
    |   3x3 pw conv   |
    |       BN        |
    |      ReLU       |
    ———————————————————
    '''
    def __init__(self, in_channels, out_channels, stride=1, padding=1, with_bn=True, conv_cfg=None, norm_cfg=None):
        super(MbnV1Module, self).__init__()

        self.with_bn = with_bn

        # BatchNorm
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, in_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, out_channels, postfix=2)
        self.add_module(self.norm1_name, norm1)
        self.add_module(self.norm2_name, norm2)

        # ReLU
        self.relu = nn.ReLU(inplace=True)

        # Depthwise Conv
        self.dw_conv = build_conv_layer(
            conv_cfg,
            in_channels,
            in_channels,
            kernel_size=3,
            groups=in_channels,
            stride=stride,
            padding=padding,
            dilation=padding,
            bias=False
        )

        # Pointwise Conv
        self.pw_conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size=1,
            groups=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False
        )

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""
        out = self.dw_conv(x)
        if self.with_bn:
            out = self.norm1(out)
        out = self.relu(out)

        out = self.pw_conv(out)
        if self.with_bn:
            out = self.norm2(out)
        out = self.relu(out)

        return out


@BACKBONES.register_module()
class MobileNetV1(nn.Module):
    CONV1_OUT_CHANNELS = 32
    
    arch_settings = {
        1: {'out_channels': (64, 128, 128, 256, 256),        'strides': (1, 2, 1, 2, 1)},
        2: {'out_channels': (512, 512, 512, 512, 512, 512),  'strides': (2, 1, 1, 1, 1, 1)},
        3: {'out_channels': (1024, 1024),                    'strides': (2, 1)}
    }

    '''
    MobileNetV1 backbone.
    '''
    def __init__(self, num_stages=3, out_indices=(0, 1, 2), width_mult=1.0, in_channels=3, with_bn=True, conv_cfg=None, norm_cfg=None):
        super(MobileNetV1, self).__init__()

        assert num_stages > 0 and num_stages <= 3
        assert max(out_indices) < num_stages

        self.num_stages = num_stages
        self.out_indices = out_indices
        self.width_mult = width_mult
        self.in_channels = in_channels
        self.with_bn = with_bn
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.conv1 = build_conv_layer(conv_cfg, in_channels, self._shrink_channel(self.CONV1_OUT_CHANNELS), kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, self._shrink_channel(self.CONV1_OUT_CHANNELS), postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)

        # self.conv1.eval()
        for param in self.conv1.parameters():
            param.requires_grad = False

        self.stages = []
        for i in range(3):
            stage = self._build_stage(i+1)
            stage_name = f'stage{i + 1}'
            self.add_module(stage_name, stage)
            self.stages.append(stage_name)

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    def _shrink_channel(self, x):
        return int(x * self.width_mult)

    def _build_stage(self, stage_idx):
        setting = self.arch_settings[stage_idx]
        out_channels = [self._shrink_channel(n) for n in setting['out_channels']]
        strides = setting['strides']
        depth = len(strides)

        if stage_idx == 1:
            out_channels.insert(0, self._shrink_channel(self.CONV1_OUT_CHANNELS))
        else:
            out_channels.insert(0, self._shrink_channel(self.arch_settings[stage_idx - 1]['out_channels'][-1]))
        
        return nn.Sequential(*[MbnV1Module(out_channels[i], out_channels[i+1], stride=strides[i], with_bn=self.with_bn, norm_cfg=self.norm_cfg) for i in range(depth)])

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)
        if self.with_bn:
            x = self.norm1(x)
        x = self.relu(x)

        outs = []
        for i, stage_name in enumerate(self.stages):
            stage = getattr(self, stage_name)
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)