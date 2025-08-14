
# © 2025 Xiaoyu Ma. 
# Code of Improving Multimodal Learning Balance and Sufficiency through Data Remixing.
# This code is adapted from OGM-GE, available at:
# https://github.com/GeWu-Lab/OGM-GE_CVPR2022
# All rights reserved.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18

class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()
        if args.dataset == 'VGGSound':
            n_classes = 309
        elif args.dataset == 'KineticSound':
            n_classes = 31
        elif args.dataset == 'CREMAD':
            n_classes = 6
        elif args.dataset == 'AVE':
            n_classes = 28
        else:
            raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))


        self.dataset = args.dataset

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

        self.head = nn.Linear(1024, n_classes)
        self.head_audio = nn.Linear(512, n_classes)
        self.head_video = nn.Linear(512, n_classes)

    def forward(self, audio, visual, mode=None):

        visual = visual.permute(0, 2, 1, 3, 4).contiguous() 
        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        if mode == "video":
            a[:, :] = 0.0

        elif mode == "audio":
            v[:, :] = 0.0

        out = torch.cat((a,v), 1)
        out = self.head(out)

        out_audio=self.head_audio(a)
        out_video=self.head_video(v)

        return out, out_audio, out_video, a, v







        
    

        
    



