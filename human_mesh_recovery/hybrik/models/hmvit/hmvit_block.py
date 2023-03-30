import math
import logging
from functools import partial
from collections import OrderedDict
# from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

class Conv(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_C, hidden_C=None, out_C=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_C = out_C or in_C
        hidden_C = hidden_C or in_C

        self.conv1 = nn.Conv2d(in_C, hidden_C, 1)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.conv2 = nn.Conv2d(hidden_C, out_C, 1)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.drop2(x)
        return x

class SELayer(nn.Module):
    def __init__(self,C, H, W, reduction=8):
        super(SELayer, self).__init__()

        self.reduce1 = nn.Linear(W, W//reduction)
        self.fc1 = nn.Linear(H*W//reduction, H*W//reduction, bias=False)
        self.up1 = nn.Linear(W//reduction, W)


        self.reduce2 = nn.Linear(H, H // reduction)
        self.fc2 = nn.Linear(H * W // reduction, H * W // reduction, bias=False)
        self.up2 = nn.Linear(H // reduction, H)
        self.ln = nn.LayerNorm([H,W])
        self.gelu = nn.GELU()

        self.conv_out = nn.Conv2d(2*C, C, 1)

    def forward(self, x):
        B, C, H, W = x.size()
        x1 = x
        x1 = self.reduce1(x1).view(B,C,-1)
        x1 = self.gelu(x1)
        x1 = self.fc1(x1).view(B,C,H,-1)
        x1 = self.up1(x1)

        x2 = x.transpose(2,3)
        x2 = self.reduce2(x2).view(B, C, -1)
        x2 = self.gelu(x2)
        x2 = self.fc2(x2).view(B, C, W, -1)
        x2 = self.up2(x2).transpose(2,3)

        x = torch.cat((x1,x2), dim = 1)
        x = self.ln(x)
        x = self.gelu(x)
        x = self.conv_out(x)

        return x

class Attention(nn.Module):
    def __init__(self, C, H, W, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_W = W // num_heads
        self.scale_W = head_W ** -0.5
        self.qkv_W = nn.Linear(W, W * 3, bias=qkv_bias)
        self.attn_drop_W = nn.Dropout(attn_drop)

        head_H = H // num_heads
        self.scale_H = head_H ** -0.5
        self.qkv_H = nn.Linear(H, H * 3, bias=qkv_bias)
        self.attn_drop_H = nn.Dropout(attn_drop)

        self.proj = nn.Conv2d(C, C, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x_H = x.transpose(2, 3)

        qkv_W = self.qkv_W(x).reshape(B, C, H, 3, self.num_heads, W // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q_W, k_W, v_W = qkv_W.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        ####  q_W  [B, num_head, num_patch, H, W//num_head]

        attn_W = (q_W.transpose(-3, -2) @ torch.permute(k_W, (0, 1, 3, 4, 2))) * self.scale_W
        attn_W = attn_W.softmax(dim=-1)
        attn_W = self.attn_drop_W(attn_W)
        x_W = (attn_W @ v_W.transpose(2, 3)).transpose(1, 3).reshape(B, C, H, W)

        qkv_H = self.qkv_H(x_H).reshape(B, C, W, 3, self.num_heads, H // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q_H, k_H, v_H = qkv_H.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn_H = (q_H.transpose(-3, -2) @ torch.permute(k_H, (0, 1, 3, 4, 2))) * self.scale_H
        attn_H = attn_H.softmax(dim=-1)
        attn_H = self.attn_drop_H(attn_H)
        x_H = (attn_H @ v_H.transpose(2, 3)).transpose(1, 3).reshape(B, C, W, H)

        # x = torch.cat((x_W, x_H.transpose(2, 3)), dim = 1)
        x = x_W + x_H.transpose(2, 3)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, C, H, W, num_heads, ratio=4, qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm([H,W])
        self.attn = Attention(C, H, W, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm([H,W])
        self.conv = Conv(in_C=C, hidden_C=int(C * ratio), act_layer=act_layer, drop=drop)


    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.conv(self.norm2(x)))

        return x


class HMVIT_block(nn.Module):


    def __init__(self, C=64, H=64, W=48, depth=12,num_heads=8,
                 ratio=4, qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., act_layer=nn.GELU, ):
        """
        """
        super().__init__()

        self.pos_embed = nn.Parameter(torch.zeros(1, C, H, W))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                C, H, W, num_heads=num_heads, ratio=ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], act_layer=act_layer)
            for i in range(depth)])

        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm([H, W])

    def forward(self, x):

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.gelu(x)
        x = self.norm(x)

        return x


