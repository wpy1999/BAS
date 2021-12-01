import torch
import os
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np 
import torch.utils.data as data
from torchvision.datasets import ImageFolder

def load_test_bbox(root, test_gt_path,crop_size,resize_size):
    test_gt = []
    test_txt = []
    shift_size = (resize_size - crop_size) // 2
    with open(test_gt_path, 'r') as f:
        
        for line in f:
            temp_gt = []
            part_1, part_2 = line.strip('\n').split(';')
            img_path, w, h, _ = part_1.split(' ')
            part_2 = part_2[1:]
            bbox = part_2.split(' ')
            bbox = np.array(bbox, dtype=np.float32)
            box_num = len(bbox) // 4
            w, h = np.float32(w),np.float32(h)
            for i in range(box_num):
                bbox[4*i], bbox[4*i+1], bbox[4*i+2], bbox[4*i+3] = bbox[4*i]/w*resize_size- shift_size, bbox[4*i+1]/h*resize_size- shift_size , bbox[4*i+2]/w*resize_size- shift_size , bbox[4*i+3]/h*resize_size- shift_size
                if bbox[4*i] < 0:
                    bbox[4*i] = 0
                if bbox[4*i+1] < 0:
                    bbox[4*i+1] = 0
                if bbox[4*i+2]>crop_size:
                    bbox[4*i+2] = crop_size
                if bbox[4*i+3]>crop_size:
                    bbox[4*i+3] = crop_size  
                temp_gt.append([bbox[4*i], bbox[4*i+1], bbox[4*i+2], bbox[4*i+3]])
            test_gt.append(temp_gt)
            img_path = img_path.replace("\\\\","\\")
            test_txt.append(img_path)
    final_dict = {}
    for k, v in zip(test_txt, test_gt):
        k = os.path.join(root, 'val', k)
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
            self.img_dataset = ImageFolder(os.path.join(self.root, 'val'))
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
        self.test_bbox = load_test_bbox(self.root, self.test_gt_path,self.crop_size,self.resize_size) 

    def __getitem__(self, index):
        path, img_class = self.img_dataset[index]
         
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
