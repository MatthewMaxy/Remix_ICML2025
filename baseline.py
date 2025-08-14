
# © 2025 Xiaoyu Ma. 
# Code of Improving Multimodal Learning Balance and Sufficiency through Data Remixing.
# This code is adapted from OGM-GE, available at:
# https://github.com/GeWu-Lab/OGM-GE_CVPR2022
# All rights reserved.

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataloader import AV_CD_Dataset
from models.models import AVClassifier
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, help='CREMAD')
    parser.add_argument('--model', default='resnet18', type=str, choices=['resnet18'])
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--n_classes', default=6, type=int)
    parser.add_argument('--encoder_lr_decay', default=1.0, type=float, help='decay coefficient')
    parser.add_argument('--optimizer', default='adam', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=30, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')
    parser.add_argument('--alpha', default=0.5, type=float, help='factor of audio-loss in warmup')
    parser.add_argument('--beta', default=0.5, type=float, help='factor of video-loss in warmup')
    parser.add_argument('--train', action='store_true', help='turn on train mode')
    parser.add_argument('--log_path', default='log_model', type=str, help='path to save model')
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--gpu_ids', default='0,1', type=str, help='GPU ids')

    return parser.parse_args()

def train_epoch(args, epoch, model, device, dataloader, optimizer):
    criterion = nn.CrossEntropyLoss()
    alpha = args.alpha
    beta = args.beta
    model.train()
    print("Start training baseline... ")

    _loss = 0
    _loss_a = 0
    _loss_v = 0

    for step, (image, spec, label, _, _) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()

        image = image.to(device)
        spec = spec.to(device)
        label = label.to(device)
        out, out_audio, out_video, a, v = model(spec.float(), image.float())

        loss_av = criterion(out, label)
        loss_a = criterion(out_audio, label)
        loss_v = criterion(out_video, label)
        loss = loss_av + alpha * loss_a + beta * loss_v
        
        loss.backward()

        optimizer.step()
        _loss += loss.item()

    return _loss / len(dataloader)
  
def valid(args, model, device, dataloader, epoch):
    
    softmax = nn.Softmax(dim=1)
    print('Testing...')
    n_classes = args.n_classes
    _loss = 0

    with torch.no_grad():
        model.eval()
        num = [0.0 for _ in range(n_classes)]
        acc = [0.0 for _ in range(n_classes)]

        for step, (image, spec, label, _, _) in enumerate(tqdm(dataloader)):
            image = image.to(device)
            spec = spec.to(device)
            label = label.to(device)
            out, out_audio, out_video, a, v = model(spec.float(), image.float())

            prediction = softmax(out)

            for i, item in enumerate(label):
                ma = prediction[i].cpu().data.numpy()
                index_ma = np.argmax(ma)
                num[label[i]] += 1.0
                if index_ma == label[i]:
                    acc[label[i]] += 1.0

    return sum(acc) / sum(num)


def main():
    args = get_arguments()
    print(args)
    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    gpu_ids = list(range(torch.cuda.device_count()))

    device = torch.device('cuda:0')


    model = AVClassifier(args)

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda()

    train_dataset = AV_CD_Dataset(mode='train')
    test_dataset = AV_CD_Dataset(mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=16,pin_memory=True)  
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=16,pin_memory=True)                                

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    best_acc = -1
    if args.train:
        for epoch in range(args.epochs):

            print('Epoch: {}: '.format(epoch))
            batch_loss = train_epoch(args, epoch, model, device, train_dataloader, optimizer)
            scheduler.step()

            acc= valid(args, model, device, test_dataloader, epoch)

            if acc > best_acc:
                best_acc = float(acc)
                model_name = 'baseline_concat_sgd.pth'
                    
                saved_dict = {'saved_epoch': epoch,
                              'acc': acc,
                              'model': model.state_dict(),
                              'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict()}

                save_dir = os.path.join(args.log_path, model_name)

                torch.save(saved_dict, save_dir)
                print('The best model has been saved at {}.'.format(save_dir))
                print("Loss: {:.4f}, Acc: {:.4f}".format(batch_loss, acc))

            else:
                print("Loss: {:.4f}, Acc: {:.4f}, Best Acc: {:.4f}".format(batch_loss, acc, best_acc))

        print(f"All training epoches of BASELINE finished, Best Acc:{best_acc}")

if __name__ == "__main__":
    main()
