"""
The code is based on https://github.com/jiaxi-jiang/FBCNN
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR',
         negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC',
                 negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res


# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R',
                          negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C' + mode,
               negative_slope=negative_slope)
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R',
                    negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR'
    if mode[0] == '2':
        uc = 'UC'
    elif mode[0] == '3':
        uc = 'uC'
    elif mode[0] == '4':
        uc = 'vC'
    mode = mode.replace(mode[0], uc)
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode, negative_slope=negative_slope)
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(in_channels=64, out_channels=3, kernel_size=2, stride=2, padding=0, bias=True, mode='2R',
                           negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'T')
    up1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return up1


# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True, mode='2R',
                          negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, bias=True, mode='2R',
                       negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:],
                     negative_slope=negative_slope)
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='2R',
                       negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3'], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode=mode[1:],
                     negative_slope=negative_slope)
    return sequential(pool, pool_tail)


class QPAttention(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC',
                 negative_slope=0.2):
        super(QPAttention, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x, gamma, beta):
        # x: b, c, h, w
        # gamma, beta: b, c
        if type(gamma) == torch.Tensor:
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
        res = gamma * self.res(x) + beta
        return x + res


class CPE(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[128, 256, 512, 1024], nb=4, act_mode='BR', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(CPE, self).__init__()

        self.m_head = conv(in_nc, nc[0], bias=True, mode='C')
        self.nb = nb
        self.nc = nc
        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = sequential(
            *[ResBlock(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=True, mode='2'))
        self.m_down2 = sequential(
            *[ResBlock(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=True, mode='4'))
        self.m_down3 = sequential(
            *[ResBlock(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=True, mode='4'))

        self.m_body_encoder = sequential(
            *[ResBlock(nc[3], nc[3], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])

        self.m_body_decoder = sequential(
            *[ResBlock(nc[3], nc[3], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = nn.ModuleList([upsample_block(nc[3], nc[2], bias=True, mode='4'),
                                    *[QPAttention(nc[2], nc[2], bias=True, mode='C' + act_mode + 'C') for _ in
                                      range(nb)]])

        self.m_up2 = nn.ModuleList([upsample_block(nc[2], nc[1], bias=True, mode='4'),
                                    *[QPAttention(nc[1], nc[1], bias=True, mode='C' + act_mode + 'C') for _ in
                                      range(nb)]])

        self.m_up1 = nn.ModuleList([upsample_block(nc[1], nc[0], bias=True, mode='2'),
                                    *[QPAttention(nc[0], nc[0], bias=True, mode='C' + act_mode + 'C') for _ in
                                      range(nb)]])

        self.m_tail = conv(nc[0], out_nc, bias=True, mode='C')

        self.shared_pred = sequential(*[ResBlock(nc[3], nc[3], bias=True, mode='C' + act_mode + 'C') for _ in range(nb)],
                                torch.nn.AdaptiveAvgPool2d((1, 1)),
                                torch.nn.Flatten(),
                                torch.nn.Linear(nc[3], nc[3]//2),
                                nn.ReLU(),
                            )  
        # Codec head 
        self.codec_head = nn.Sequential(
            torch.nn.Linear(nc[3]//2, 4) 
        )
        # Codec embedding
        self.codec_embed = nn.Embedding(4, 32)

        # QP head 
        self.qp_head = nn.Sequential(
            nn.Linear(nc[3]//2 + 32, 128),  # shared_feat + codec_emb
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )



    def forward(self, x):

        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body_encoder(x4)
        
        shared_feat = self.shared_pred(x)              # (B, 512)

        codec_logit = self.codec_head(shared_feat)     # (B, 4)

        codec_prob  = torch.softmax(codec_logit, dim=1)  # (B, 4)
        codec_emb   = codec_prob @ self.codec_embed.weight  # (B, 32)

        qp_input = torch.cat([shared_feat, codec_emb], dim=1)  # (B, 512+32)
        qp       = self.qp_head(qp_input)             # (B, 1)

        # decoder
        x = self.m_body_decoder(x)

        gamma_1, beta_1, gamma_2, beta_2, gamma_3, beta_3 = 1., 1., 1., 1., 1., 1.

        x = x + x4
        x = self.m_up3[0](x)
        for i in range(self.nb):
            x = self.m_up3[i + 1](x, gamma_3, beta_3)
        x = x + x3

        x = self.m_up2[0](x)
        for i in range(self.nb):
            x = self.m_up2[i + 1](x, gamma_2, beta_2)
        x = x + x2

        x = self.m_up1[0](x)
        for i in range(self.nb):
            x = self.m_up1[i + 1](x, gamma_1, beta_1)

        x = x + x1
        x = self.m_tail(x)
        x = x[..., :h, :w]

        return x, qp, codec_logit

    def get_visual_embedding(self, x):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)  # b, 3, h, w
        x1 = self.m_head(x)  # b, 64, h, w
        x2 = self.m_down1(x1)  # b, 128, h // 2, w // 2
        x3 = self.m_down2(x2)  # b, 256, h // 4, w // 4
        x4 = self.m_down3(x3)  # b, 512, h // 8, w // 8
        x = self.m_body_encoder(x4)  # b, 512, h // 8, w // 8
        x = torch.nn.Flatten(-2, -1)(x)
        x = x.permute(0, 2, 1)
        # qp = self.qp_pred(x)  # b, 1

        return x

    def pred_qp(self, x):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)  # b, 3, h, w
        x1 = self.m_head(x)  # b, 64, h, w
        x2 = self.m_down1(x1)  # b, 128, h // 2, w // 2
        x3 = self.m_down2(x2)  # b, 256, h // 4, w // 4
        x4 = self.m_down3(x3)  # b, 512, h // 8, w // 8
        x = self.m_body_encoder(x4)  # b, 512, h // 8, w // 8
        qp = self.qp_pred(x)  # b, 1
        return qp
