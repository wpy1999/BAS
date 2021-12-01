import torch
import os
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np 
import torch.utils.data as data
from torchvision.datasets import ImageFolder

def load_test_bbox(root, test_txt_path, test_gt_path,resize_size, crop_size):
    test_gt = []
    test_txt = []
    shift_size = (resize_size - crop_size) // 2.
    with open(test_txt_path, 'r') as f:
        for line in f:
            img_path = line.strip('\n').split(';')[0]
            test_txt.append(img_path)
    with open(test_gt_path, 'r') as f:
        
        for line in f:
            x0, y0, x1, y1, h, w = line.strip('\n').split(' ')
            x0, y0, x1, y1, h, w = float(x0), float(y0), float(x1), float(y1), float(h), float(w)
            x0, y0, x1, y1 = x0/w*resize_size - shift_size, y0/h*resize_size - shift_size, x1/w*resize_size - shift_size, y1/h*resize_size - shift_size
            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            if x1>crop_size:
                x1 = crop_size
            if y1>crop_size:
                y1 = crop_size  
            test_gt.append([x0, y0, x1, y1])
    final_dict = {}
    for k, v in zip(test_txt, test_gt):
        k = os.path.join(root, 'test', k)
        k = k.replace('/', '\\')
        final_dict[k] = v
    return final_dict

class ImageDataset(data.Dataset):
    def __init__(self, args):
        self.args =args
        self.root = args.root
        self.test_txt_path = args.test_txt_path
        self.test_gt_path = args.test_gt_path
        self.crop_size = args.crop_size 
        self.resize_size = args.resize_size
        self.phase = args.phase
        self.num_classes = args.num_classes
        if self.phase == 'train':
            self.img_dataset = ImageFolder(os.path.join(self.root, 'train'))
            self.transform = transforms.Compose([
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        elif self.phase == 'test':
            self.img_dataset = ImageFolder(os.path.join(self.root, 'test'))
            self.transform = transforms.Compose([
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        
        self.label_classes = []  
        for k, v in self.img_dataset.class_to_idx.items():
            self.label_classes.append(k)
        self.img_dataset = self.img_dataset.imgs   
        self.test_bbox = load_test_bbox(self.root, self.test_txt_path, self.test_gt_path,self.resize_size, self.crop_size)  

    def __getitem__(self, index):
        path, img_class = self.img_dataset[index]
        ##  one-hot 
        label = torch.zeros(self.num_classes)
        label[img_class] = 1
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        if self.phase == 'train':
            return path, img, img_class
        else:
            path = path.replace('/', '\\')
            bbox = self.test_bbox[path]
            return img, img_class, bbox, path

    def __len__(self):
        return len(self.img_dataset)