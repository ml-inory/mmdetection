import logging
from mmcv.runner import load_checkpoint
import torch
import torch.nn as nn
from mmcv.cnn import (constant_init, kaiming_init, normal_init)
from ..builder import BACKBONES

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_bn(inp, oup, stride, groups=1, act_fn=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False, groups=groups),
        nn.BatchNorm2d(oup),
        act_fn(inplace=True)
    )


def conv_1x1_bn(inp, oup, groups=1, act_fn=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False, groups=groups),
        nn.BatchNorm2d(oup),
        act_fn(inplace=True)
    )

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@BACKBONES.register_module()
class SSDMobilenetV2(nn.Module):
    def __init__(self,
                 input_size,
                 activation_type,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 out_indices=(1, 3, 6, 10, 13, 16, 17)):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(SSDMobilenetV2, self).__init__()
        self.input_size = input_size
        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 320

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s, output
                [1, 16, 1, 1, False],
                [6, 24, 2, 2, False],
                [6, 32, 3, 2, False],
                [6, 64, 4, 2, False],
                [6, 96, 3, 1, False],
                [6, 160, 3, 2, True],
                [6, 320, 1, 1, False],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 5:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 5-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.out_indices = out_indices
        self.output = [False]
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s, o in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
                if o and i == n-1:
                    self.output.append(True)
                else:
                    self.output.append(False)
            # building last several layers
        # if len(features) in out_indices:
        #     features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        #     self.output.append(True)

        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        
        # building last several layers
        self.extra_convs = []
        self.extra_convs.append(ConvBNReLU(last_channel, 1280, 1, 1))
        self.extra_convs.append(ConvBNReLU(1280, 256, 1, 1))
        self.extra_convs.append(ConvBNReLU(256, 256, 3, 2, groups=256))
        self.extra_convs.append(ConvBNReLU(256, 512, 1, 1))
        self.extra_convs.append(ConvBNReLU(512, 128, 1, 1))
        self.extra_convs.append(ConvBNReLU(128, 128, 3, 2, groups=128))
        self.extra_convs.append(ConvBNReLU(128, 256, 1, 1))
        self.extra_convs.append(ConvBNReLU(256, 128, 1, 1))
        self.extra_convs.append(ConvBNReLU(128, 128, 3, 2, groups=128))
        self.extra_convs.append(ConvBNReLU(128, 256, 1, 1))
        self.extra_convs.append(ConvBNReLU(256, 64, 1, 1))
        self.extra_convs.append(ConvBNReLU(64, 64, 3, 2, groups=64))
        self.extra_convs.append(ConvBNReLU(64, 128, 1, 1))

        self.extra_convs = nn.Sequential(*self.extra_convs)

    def init_weights(self, pretrained):
        # weight initialization
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        out = []
        for layer, output in zip(self.features, self.output):
            x = layer(x)
            if output:
                out.append(x)
                
        for i, conv in enumerate(self.extra_convs):
            x = conv(x)
            if i % 3 == 0:
                out.append(x)

        return out


    def forward(self, x):
        return self._forward_impl(x)

    #＃　freeze first stage or not
    def train(self, mode=True):
        super(SSDMobilenetV2, self).train(mode)
    #     self._freeze_stages()
    #     if mode and self.norm_eval:
    #         for m in self.modules():
    #             # trick: eval have effect on BatchNorm only
    #             if isinstance(m, _BatchNorm):
    #                 m.eval()