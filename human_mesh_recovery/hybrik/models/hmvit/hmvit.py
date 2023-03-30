# ------------------------------------------------------------------------------
# Copyright (c) Southeast University. Licensed under the MIT License.
# Written by Sen Yang (yangsenius@seu.edu.cn)
# ------------------------------------------------------------------------------

import os

import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .hmvit_block import HMVIT_block
from .hr_base import HRNET_base

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_C, out_C, H, W, stride=1,):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_C, out_C, stride)
        self.ln1 = nn.LayerNorm([H, W])
        self.gelu = nn.GELU()
        self.conv2 = conv3x3(in_C, out_C,)
        self.ln2 = nn.LayerNorm([H, W])
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.ln1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.ln2(out)

        out += residual
        out = self.gelu(out)

        return out

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_H=256, img_W=192, in_C=3, patch_size = 4, out_C = 64):
        super().__init__()
        self.out_H = img_H // patch_size
        self.out_W = img_W // patch_size
        channel = in_C * patch_size * patch_size
        self.proj = nn.Conv2d(channel, out_C, kernel_size=1)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm([self.out_H, self.out_W])
        # self.norm = nn.BatchNorm2d(out_C)
        self.layer1 = BasicBlock(out_C, out_C, self.out_H, self.out_W,)
        self.layer2 = BasicBlock(out_C, out_C, self.out_H, self.out_W, )
        self.layer3 = BasicBlock(out_C, out_C, self.out_H, self.out_W, )
        self.layer4 = BasicBlock(out_C, out_C, self.out_H, self.out_W, )

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1, self.out_H, self.out_W)
        x = self.proj(x)
        x = self.gelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.norm(x)

        return x


class HMVIT(nn.Module):

    def __init__(self, cfg, img_H=256, img_W=192, patch_size = 4, out_C = 32, depth = 8, num_joints=17,
                 ratio=8, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, **kwargs):

        super(HMVIT, self).__init__()

        self.pre_feature = HRNET_base(cfg, **kwargs)

        out_H = img_H // patch_size
        out_W = img_W // patch_size
        self.CONVIT_Block = HMVIT_block(C=out_C, H=out_H, W=out_W, depth=depth, ratio=ratio,
                                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, )


    def forward(self, x):
        x = self.pre_feature(x)
        x_feature = self.CONVIT_Block(x)


        return x_feature

    def init_weights(self, pretrained=''):
        self.pre_feature.init_weights(pretrained)


def load_pretrained_weights(model, checkpoint):
    import collections
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith('module.'):
            k = k[7:]
        if k.startswith('backbone.'):
            k = k[9:]
        if k.startswith('hrt_back.'):
            k = k[9:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print('load_weight from hmvit', len(matched_layers))
    return model

cfg=dict(   MODEL = dict(
        EXTRA =  dict(
            PRETRAINED_LAYERS  = ('conv1','bn1','conv2','bn2','layer1','transition1','stage2','transition2','stage3',),
            FINAL_CONV_KERNEL=1,
            STAGE2=dict(
                NUM_MODULES=1,
                NUM_BRANCHES=2,
                BLOCK='BASIC',
                NUM_BLOCKS=(4, 4),
                NUM_CHANNELS=(32, 64),
                FUSE_METHOD="SUM",),
            STAGE3=dict(
                NUM_MODULES=4,
                NUM_BRANCHES=3,
                BLOCK='BASIC',
                NUM_BLOCKS=(4, 4, 4),
                NUM_CHANNELS=(32, 64, 128),
                FUSE_METHOD="SUM",)
            )))

def get_pose_net(num_joints, img_H, img_W, checkpoint, **kwargs):
    # num_joints = cfg.MODEL.NUM_JOINTS,
    # img_H = cfg.MODEL.IMAGE_SIZE[1]
    # img_W = cfg.MODEL.IMAGE_SIZE[0]

    model = HMVIT(cfg, img_H=img_H, img_W=img_W, patch_size = 4, out_C = 32, depth = 8, num_joints=num_joints, **kwargs)

    # if is_train and cfg.MODEL.INIT_WEIGHTS:
    #     model.init_weights(cfg.MODEL.PRETRAINED)
    if checkpoint is not None:
        hmvit_checkpoint = torch.load(checkpoint,
                                      map_location=lambda storage, loc: storage)
        if "state_dict" in hmvit_checkpoint.keys():
            hmvit_checkpoint = hmvit_checkpoint["state_dict"]
        model = load_pretrained_weights(model, hmvit_checkpoint)

    return model




