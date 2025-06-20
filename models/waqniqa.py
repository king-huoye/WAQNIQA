import torch
import torch.nn as nn
import timm

from timm.models.vision_transformer import Block
from models.swin import SwinTransformer
from torch import nn
from einops import rearrange
import pywt
from base.quaternion_layers import QuaternionConv, QuaternionTransposeConv




# 这是可以跑通的小波变换

import torch
import torch.nn as nn
import pywt
import numpy as np


class WIAA(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def wavelet_transform(self, x):
        # 假设 x 的形状是 (B, C, N)
        B, C, N = x.shape
        wavelet = 'haar'


        x_np = x.detach().cpu().numpy()
        transformed = np.zeros_like(x_np)

        for b in range(B):
            for c in range(C):

                coeffs = pywt.wavedec(x_np[b, c], wavelet, level=1)


                coeffs_array = coeffs[0]


                if len(coeffs_array) < N:

                    pad_size = N - len(coeffs_array)
                    coeffs_array = np.pad(coeffs_array, (0, pad_size), mode='constant')
                elif len(coeffs_array) > N:

                    coeffs_array = coeffs_array[:N]

                transformed[b, c] = coeffs_array


        return torch.tensor(transformed, dtype=x.dtype, device=x.device)

    def forward(self, x):
        _x = x
        B, C, N = x.shape

        x = self.wavelet_transform(x)

        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x

        return x


class QCFE(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, relu=False, transpose=False):
        super(QCFE, self).__init__()
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(QuaternionTransposeConv(in_channel, out_channel, kernel_size, padding=padding, stride=stride,
                                                  bias=bias))

        else:
            layers.append(
                QuaternionConv(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))

        if relu:
            layers.append(nn.GELU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


class WAQNIQA(nn.Module):
    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1,
                 depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                 img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)

        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = WIAA(self.input_size ** 2)
            self.tablock1.append(tab)


        self.conv1 = QCFE(embed_dim * 4, embed_dim, 1, 1, True)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = WIAA(self.input_size ** 2)
            self.tablock2.append(tab)

        self.conv2 = QCFE(embed_dim, embed_dim // 2, 1, 1, True)



        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )

    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    def forward(self, x):
        _x = self.vit(x)
        x = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()

        # stage 1
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        x = self.swintransformer1(x)

        # stage2
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x = self.swintransformer2(x)

        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score
