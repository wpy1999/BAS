import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 
import cv2
from skimage import measure
from utils.func import *

class Model(nn.Module): 
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_classes = args.num_classes 
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  ## -> 64x224x224
        self.relu1_1 = nn.ReLU(inplace=True) 
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1) ## -> 64x224x224
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)  ## -> 64x112x112

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  ## -> 128x112x112
        self.relu2_1 = nn.ReLU(inplace=True) 
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1) ## -> 128x112x112
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2) ## -> 128x56x56
        
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1) ## -> 256x56x56
        self.relu3_1 = nn.ReLU(inplace=True) 
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1) ## -> 256x56x56
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1) ## -> 256x56x56
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2) ## -> 256x28x28 

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1) ## -> 512x28x28
        self.relu4_1 = nn.ReLU(inplace=True) 
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1) ## -> 512x28x28
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1) ## -> 512x28x28
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2) ## -> 512x14x14

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1) ## -> 512x14x14
        self.relu5_1 = nn.ReLU(inplace=True) 
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1) ## -> 512x14x14
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1) ## -> 512x14x14
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2) ## -> 512x7x7

        self.avg_pool = nn.AvgPool2d(14) ## ->(512,1,1)

        self.classifier_cls = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1000, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.classifier_loc = nn.Sequential( 
            nn.Conv2d(512, 1000, kernel_size=3, padding=1),
            nn.Sigmoid(),
        ) 
        self.conv_copy_5_1 = copy.deepcopy(self.conv5_1)   
        self.conv_copy_5_2 = copy.deepcopy(self.conv5_2)   
        self.conv_copy_5_3 = copy.deepcopy(self.conv5_3)  
        self.classifier_cls_copy = copy.deepcopy(self.classifier_cls)

    def forward(self, x, label=None,N=1):
        self.weight_deepcopy()
        batch = x.size(0)

        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x = self.pool3(x)
        
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)
        x_4 = x.clone() 
        
        x = self.pool4(x)
        
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)

        x = self.classifier_cls(x)
        self.feature_map = x

## score
        x = self.avg_pool(x).view(x.size(0), -1)
        self.score_1 = x    

# p_label 
        if N == 1:
            p_label = label.unsqueeze(-1)
        else:
            _, p_label = self.score_1.topk(N, 1, True, True)

# x_sum   
        self.x_sum = torch.zeros(batch).cuda()
        for i in range(batch):
            self.x_sum[i] = self.score_1[i][label[i]]

## x_saliency    
        x_saliency_all = self.classifier_loc(x_4)
        x_saliency =  torch.zeros(batch, 1, 28, 28).cuda()
        for i in range(batch):
            x_saliency[i][0] = x_saliency_all[i][p_label[i]].mean(0)
        self.x_saliency = x_saliency

## erase
        x_erase = x_4.detach() * ( 1 - x_saliency)        
        x_erase = self.pool4(x_erase)
        x_erase = self.conv_copy_5_1(x_erase)
        x_erase = self.relu5_1(x_erase)
        x_erase = self.conv_copy_5_2(x_erase)
        x_erase = self.relu5_2(x_erase)
        x_erase = self.conv_copy_5_3(x_erase)
        x_erase = self.relu5_3(x_erase)
        x_erase = self.classifier_cls_copy(x_erase)
        x_erase = self.avg_pool(x_erase).view(x_erase.size(0), -1)

## x_erase_sum
        self.x_erase_sum = torch.zeros(batch).cuda()
        for i in range(batch):
            self.x_erase_sum[i] = x_erase[i][label[i]]

##  score_2 
        x = self.feature_map * nn.AvgPool2d(2)(self.x_saliency)
        self.score_2 = self.avg_pool(x).squeeze(-1).squeeze(-1)        

##  loss      
        x_sum = self.x_sum.clone().detach()
        x_res = self.x_erase_sum
        res = x_res / (x_sum+1e-8)
        res[x_res>x_sum] = 0
        x_saliency =  x_saliency.clone().view(batch, -1)
        x_saliency = x_saliency.mean(1)  
        loss_loc = res + x_saliency 
## loss 
        loss_loc = loss_loc.mean(0) 

##  loss_cls 
        loss_fnc = nn.CrossEntropyLoss()
        loss_cls_1 = loss_fnc(self.score_1, label).cuda()
        loss_cls_2 = loss_fnc(self.score_2, label).cuda()
        loss_cls = loss_cls_1 + loss_cls_2 * 0.05
        return self.score_1, loss_cls , loss_loc 
    
    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        #--------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins + 1e-10)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed
    
    def weight_deepcopy(self):
        self.conv_copy_5_1.weight.data = self.conv5_1.weight.clone().detach()
        self.conv_copy_5_2.weight.data = self.conv5_2.weight.clone().detach()
        self.conv_copy_5_3.weight.data = self.conv5_3.weight.clone().detach()
        self.conv_copy_5_1.bias.data = self.conv5_1.bias.clone().detach()
        self.conv_copy_5_2.bias.data = self.conv5_2.bias.clone().detach()
        self.conv_copy_5_3.bias.data = self.conv5_3.bias.clone().detach()
        for i in range(len(self.classifier_cls)):
            if 'Conv' in str(self.classifier_cls[i]) or 'BatchNorm2d' in str(self.classifier_cls[i]):
                self.classifier_cls_copy[i].weight.data = self.classifier_cls[i].weight.clone().detach()
                self.classifier_cls_copy[i].bias.data = self.classifier_cls[i].bias.clone().detach()

        
def weight_init(m):
    classname = m.__class__.__name__   
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)    
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.weight.data.fill_(0) 

def model(args, pretrained=True):
    model = Model(args)
    model.apply(weight_init)   
    if pretrained:
        pretrained_dict = torch.load('vgg16.pth') 
        model_dict = model.state_dict()  
        model_conv_name = []

        for i, (k, v) in enumerate(model_dict.items()):
            model_conv_name.append(k)
        for i, (k, v) in enumerate(pretrained_dict.items()):
            if k.split('.')[0] != 'features':
                break
            if np.shape(model_dict[model_conv_name[i]]) == np.shape(v):
                model_dict[model_conv_name[i]] = v 
        model.load_state_dict(model_dict)
        print("pretrained weight load complete..")
    return model
