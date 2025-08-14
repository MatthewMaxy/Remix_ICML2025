
# © 2025 Xiaoyu Ma. 
# Code of Improving Multimodal Learning Balance and Sufficiency through Data Remixing.
# This code is adapted from OGM-GE, available at:
# https://github.com/GeWu-Lab/OGM-GE_CVPR2022
# All rights reserved.

import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import random
import copy
import csv

class AV_CD_Dataset(Dataset):
    def __init__(self, mode='train'):
        classes = []
        self.data = []
        data2class = {}

        self.mode=mode
        self.visual_path = '/data3/PublicData/CREMAD_Frame_Audio/visual/'
        self.audio_path = '/data3/PublicData/CREMAD_Frame_Audio/audio/'
        self.stat_path = './data/stat.csv'
        self.train_txt = './data/train.csv'
        self.test_txt = './data/test.csv'

        if mode == 'train' or mode == 'val':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(self.stat_path, encoding='UTF-8-sig') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])
        
        with open(csv_file) as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                if item[1] in classes and os.path.exists(self.audio_path + item[0] + '.npy') and os.path.exists(
                                self.visual_path + item[0]):
                    self.data.append(item[0])
                    data2class[item[0]] = item[1]
        
        self.classes = sorted(classes)
        self.data2class = data2class
        self._init_atransform()

        print('data load over')
        print('# of files = %d ' % len(self.data))
        print('# of classes = %d' % len(self.classes))

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

  
    def __getitem__(self, idx):
        # datum: file name without .xxx
        datum = self.data[idx]

        # Audio
        spectrogram = np.load(os.path.join(self.audio_path, datum + '.npy'))
        spectrogram = np.expand_dims(spectrogram, axis=0)

        # Visual
        if self.mode == 'train':
            transf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        else:
            transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        folder_path = os.path.join(self.visual_path, datum)
        file_num = len(os.listdir(folder_path))
        pick_num = 2
        seg = int(file_num/pick_num)

        for i in range(pick_num):
            if self.mode == 'train':
                # Ensure selected index <= the largest index
                start_index = i * seg
                end_index = min((i + 1) * seg - 1, file_num - 1)  
                index = random.randint(start_index, end_index)
            else:
                index = min(i * seg + int(seg / 2), file_num - 1)
                
            path = os.path.join(folder_path, 'frame_0000' + str(index+1) + '.jpg')
            image_arr = transf(Image.open(path).convert('RGB')).unsqueeze(1).float()

            if i == 0:
                image_n = copy.copy(image_arr)
            else:
                image_n = torch.cat((image_n, image_arr), 1)
        
        # image, audio, classindex, classname, filename 
        return image_n, spectrogram, self.classes.index(self.data2class[datum]), self.data2class[datum], datum


class AV_CD_Dataset_Remix(Dataset):
    def __init__(self, modality=None):

        classes = []
        self.data = []
        data2class = {}

        self.modality = modality
        self.visual_path = '/data3/PublicData/CREMAD_Frame_Audio/visual/'
        self.audio_path = '/data3/PublicData/CREMAD_Frame_Audio/audio/'
        self.stat_path = './data/stat.csv'
        self.audio_specific_train_txt = './data/remix_a.csv'
        self.video_specific_train_txt = './data/remix_v.csv'

        if modality == 'audio':
            csv_file = self.audio_specific_train_txt
        else:
            csv_file = self.video_specific_train_txt

        with open(self.stat_path, encoding='UTF-8-sig') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])
        
        with open(csv_file) as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                if item[1] in classes and os.path.exists(self.audio_path + item[0] + '.npy') and os.path.exists(
                                self.visual_path + item[0]):
                    self.data.append(item[0])
                    data2class[item[0]] = item[1]

        self.classes = sorted(classes)
        self.data2class = data2class
        self._init_atransform()

        print(f'{modality} data load over')
        print('# of files = %d ' % len(self.data))
        print('# of classes = %d' % len(self.classes))


    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):

        # Same as normal dataset, modality masking will be adopted in model.forward()
        datum = self.data[idx]

        # Audio
        spectrogram = np.load(os.path.join(self.audio_path, datum + '.npy'))
        spectrogram = np.expand_dims(spectrogram, axis=0)

        # Visual
        transf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        folder_path = os.path.join(self.visual_path, datum)
        file_num = len(os.listdir(folder_path))
        pick_num = 2
        seg = int(file_num/pick_num)

        for i in range(pick_num):
            start_index = i * seg
            end_index = min((i + 1) * seg - 1, file_num - 1) 
            index = random.randint(start_index, end_index)
                
            path = os.path.join(folder_path, 'frame_0000' + str(index+1) + '.jpg')
            image_arr = transf(Image.open(path).convert('RGB')).unsqueeze(1).float()

            if i == 0:
                image_n = copy.copy(image_arr)
            else:
                image_n = torch.cat((image_n, image_arr), 1)

        # image, audio, classindex, classname, filename 
        return image_n, spectrogram, self.classes.index(self.data2class[datum]), self.data2class[datum], datum
