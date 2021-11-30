import os
import sys
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
import torch.nn as nn
import torchvision
from PIL import Image
import cv2
from utils.func import *
from utils.vis import *
from utils.IoU import *
from utils.augment import *
import argparse
from Model import *
from skimage import measure

parser = argparse.ArgumentParser(description='Parameters for BAS evaluation')
parser.add_argument('--input_size',default=256,dest='input_size')
parser.add_argument('--crop_size',default=224,dest='crop_size')
parser.add_argument('--num_classes',default=200)
parser.add_argument('--tencrop', default=True)
parser.add_argument('--phase', type=str, default='test')  
parser.add_argument('--gpu',help='which gpu to use',default='0',dest='gpu')
parser.add_argument('--data',metavar='DIR',default='CUB_200_2011/',help='path to imagenet dataset')
parser.add_argument('--threshold', type=float, default=0.15)
parser.add_argument('--top_k', type=int, default=200)
parser.add_argument('--arch', type=str, default='resnet_ours')  ## choosen  [ vgg_ours, resnet_ours, inceptionv3_ours, mobilenet_ours]        
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if args.arch == 'vgg_ours':
    args.top_k = 80
if args.arch == 'inceptionv3_ours':
    args.threshold = 0.1

def normalize_map(atten_map,w,h):
    min_val = np.min(atten_map)
    max_val = np.max(atten_map)
    atten_norm = (atten_map - min_val)/(max_val - min_val)
    atten_norm = cv2.resize(atten_norm, dsize=(w,h))
    return atten_norm
def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data  
    
cudnn.benchmark = True
TEN_CROP = args.tencrop
normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
transform = transforms.Compose([
        transforms.Resize((args.input_size,args.input_size)),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize
])
cls_transform = transforms.Compose([
        transforms.Resize((args.input_size,args.input_size)),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        normalize
])
ten_crop_aug = transforms.Compose([
    transforms.Resize((args.input_size, args.input_size)),
    transforms.TenCrop(args.crop_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
])
model = eval(args.arch).model(args)
model.load_state_dict(torch.load('logs/' + args.arch +'/'+ args.arch  + '99.pth.tar'))

model = model.to(0)
model.eval()
root = args.data
val_imagedir = os.path.join(root, 'test')

anno_root = os.path.join(root,'bbox')
val_annodir = os.path.join(root, 'test_gt.txt')
val_list_path = os.path.join(root, 'test_list.txt')

classes = os.listdir(val_imagedir)
classes.sort()
temp_softmax = nn.Softmax()

class_to_idx = {classes[i]:i for i in range(len(classes))}

result = {}

accs = []
accs_top5 = []
loc_accs = []
cls_accs = []
cls_5_accs = []
final_cls = []
final_loc = []
final_clsloc = []
final_clsloctop5 = []
bbox_f = open(val_annodir, 'r')
bbox_list = []
for line in bbox_f:
    x0, y0, x1, y1, h, w = line.strip("\n").split(' ')
    x0, y0, x1, y1, h, w = float(x0), float(y0), float(x1), float(y1), float(h), float(w)
    x0, y0, x1, y1 = x0, y0, x1, y1
    bbox_list.append((x0, y0, x1, y1))  ## gt
cur_num = 0
bbox_f.close()

files = [[] for i in range(200)]  

with open(val_list_path, 'r') as f:
    for line in f:
        test_img_path, img_class =  line.strip("\n").split(';')
        files[int(img_class)].append(test_img_path)

for k in range(200):
    cls = classes[k]

    total = 0
    IoUSet = []
    IoUSetTop5 = []
    LocSet = []
    ClsSet = []
    ClsSet_5 = []


    for (i, name) in enumerate(files[k]):

        gt_boxes = bbox_list[cur_num]
        cur_num += 1
        if len(gt_boxes)==0:
            continue

        raw_img = Image.open(os.path.join(val_imagedir, name)).convert('RGB')
        w, h = args.crop_size, args.crop_size

        with torch.no_grad():
            img = transform(raw_img)
            img = torch.unsqueeze(img, 0)
            img = img.to(0)
            reg_outputs = model(img,torch.tensor([class_to_idx[cls]]),args.top_k)
            
            cam = model.x_saliency[0][0].data.cpu()
            cam = normalize_map(np.array(cam),w,h)
            
            highlight = np.zeros(cam.shape)
            highlight[cam > args.threshold] = 1
            # max component
            all_labels = measure.label(highlight)
            highlight = np.zeros(highlight.shape)
            highlight[all_labels == count_max(all_labels.tolist())] = 1
            highlight = np.round(highlight * 255)
            highlight_big = cv2.resize(highlight, (w, h), interpolation=cv2.INTER_NEAREST)
            CAMs = copy.deepcopy(highlight_big)
            props = measure.regionprops(highlight_big.astype(int))

            if len(props) == 0:
                bbox = [0, 0, w, h]
            else:
                temp = props[0]['bbox']
                bbox = [temp[1], temp[0], temp[3], temp[2]]

            if TEN_CROP:
                img = ten_crop_aug(raw_img)
                img = img.to(0)
                vgg16_out = model(img,torch.tensor(class_to_idx[cls]).expand(10))
                vgg16_out = nn.Softmax()(vgg16_out)
                vgg16_out = torch.mean(vgg16_out,dim=0,keepdim=True)
                vgg16_out = torch.topk(vgg16_out, 5, 1)[1]
            else:
                img = cls_transform(raw_img)
                img = torch.unsqueeze(img, 0)
                img = img.to(0)
                vgg16_out = model(img,[class_to_idx[cls]])
                vgg16_out = torch.topk(vgg16_out, 5, 1)[1]
            vgg16_out = to_data(vgg16_out)
            vgg16_out = torch.squeeze(vgg16_out)
            vgg16_out = vgg16_out.numpy()
            out = vgg16_out
        ClsSet.append(out[0]==class_to_idx[cls])

        #handle resize and centercrop for gt_boxes

        gt_bbox_i = list(gt_boxes)
        raw_img_i = raw_img
        raw_img_i, gt_bbox_i = ResizedBBoxCrop((256,256))(raw_img, gt_bbox_i)
        raw_img_i, gt_bbox_i = CenterBBoxCrop((224))(raw_img_i, gt_bbox_i)
       # w, h = raw_img_i.size
        gt_bbox_i[0] = gt_bbox_i[0] * w
        gt_bbox_i[2] = gt_bbox_i[2] * w
        gt_bbox_i[1] = gt_bbox_i[1] * h
        gt_bbox_i[3] = gt_bbox_i[3] * h

        gt_boxes = gt_bbox_i
        #print(bbox, [int(gt_boxes[0]),int(gt_boxes[1]),int(gt_boxes[2]),int(gt_boxes[3])], name)

        bbox[0] = bbox[0]  
        bbox[2] = bbox[2] 
        bbox[1] = bbox[1] 
        bbox[3] = bbox[3]  
        #print(gt_bbox_i, bbox)
        max_iou = -1
        iou = IoU(bbox, gt_boxes)
        if iou > max_iou:
            max_iou = iou

        LocSet.append(max_iou)
        temp_loc_iou = max_iou
        if out[0] != class_to_idx[cls]:
            max_iou = 0

        result[os.path.join(cls, name)] = bbox   #max_iou
        IoUSet.append(max_iou)
        #cal top5 IoU
        max_iou = 0
        max_cls = 0
        for i in range(5):
            if out[i] == class_to_idx[cls]:
                max_iou = temp_loc_iou
                max_cls = 1
        IoUSetTop5.append(max_iou)
        ClsSet_5.append(max_cls)
        #visualization code
    cls_loc_acc = np.sum(np.array(IoUSet) > 0.5) / len(IoUSet)
    final_clsloc.extend(IoUSet)
    cls_loc_acc_top5 = np.sum(np.array(IoUSetTop5) > 0.5) / len(IoUSetTop5)
    final_clsloctop5.extend(IoUSetTop5)
    loc_acc = np.sum(np.array(LocSet) > 0.5) / len(LocSet)
    final_loc.extend(LocSet)
    cls_acc = np.sum(np.array(ClsSet))/len(ClsSet)
    cls_5_acc = np.sum(np.array(ClsSet_5))/len(ClsSet_5)
    final_cls.extend(ClsSet)
    print('{} cls-loc acc is {}, loc acc is {}, cls acc is {}'.format(cls, cls_loc_acc, loc_acc, cls_acc))
    with open('inference_CorLoc.txt', 'a+') as corloc_f:
        corloc_f.write('{} {}\n'.format(cls, loc_acc))
    accs.append(cls_loc_acc)
    accs_top5.append(cls_loc_acc_top5)
    loc_accs.append(loc_acc)
    cls_accs.append(cls_acc)
    cls_5_accs.append(cls_5_acc)
    if (k+1) %100==0:
        print(k)

print(accs)
print('Cls-Loc acc {}'.format(np.mean(accs)))
print('Cls-Loc acc Top 5 {}'.format(np.mean(accs_top5)))

print('GT Loc acc {}'.format(np.mean(loc_accs)))
print('{} cls acc {}'.format(args.arch, np.mean(cls_accs)))
print('{} cls_5_acc {}'.format(args.arch, np.mean(cls_5_accs)))
with open('Corloc_result.txt', 'w') as f:
    for k in sorted(result.keys()):
        f.write('{} {}\n'.format(k, str(result[k])))
