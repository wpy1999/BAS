import torch
import torch.nn as nn 
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 
import cv2
from skimage import measure
from utils.func import *
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        self.model = nn.Sequential(
            conv_bn(  3,  32, 2),  ## ->112
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),  ## ->56
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),  ## ->28
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 1),   
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1), 
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),  ## ->14
            conv_dw(1024, 1024, 1),
        )
        self.avg_pool = nn.AvgPool2d(14)  
        self.classifier_cls = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 200, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),  
        )
        self.classifier_loc = nn.Sequential( 
            nn.Conv2d(512, 200, kernel_size=3, padding=1),
            nn.Sigmoid(),
        ) 
        self._initialize_weights()

    def forward(self, x, label=None, N=1):
        erase_branch = copy.deepcopy(self.model[-2:])
        classifier_cls_copy = copy.deepcopy(self.classifier_cls)
        
        batch = x.size(0)
        x = self.model[:-2](x)
        x_4 = x.clone()
        x = self.model[-2:](x)
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
        x_saliency = torch.zeros(batch, 1, 28, 28).cuda()
        for i in range(batch):
            x_saliency[i][0] = x_saliency_all[i][p_label[i]].mean(0)
        self.x_saliency = x_saliency

## erase
        x_erase = x_4.detach() * ( 1 - x_saliency)
        x_erase = erase_branch(x_erase)
        x_erase = classifier_cls_copy(x_erase)
        x_erase = self.avg_pool(x_erase).view(x_erase.size(0), -1) 

## x_erase_sum
        self.x_erase_sum = torch.zeros(batch).cuda()
        for i in range(batch):
            self.x_erase_sum[i] = x_erase[i][label[i]]

## score_2
        x = self.feature_map * nn.AvgPool2d(2)(self.x_saliency)
        self.score_2 = self.avg_pool(x).squeeze(-1).squeeze(-1)

        return self.score_1, self.score_2 

    def bas_loss(self):
        batch = self.x_sum.size(0)
        x_sum = self.x_sum.clone().detach()
        x_res = self.x_erase_sum
        res = x_res / (x_sum + 1e-8)
        res[x_res>=x_sum] = 0 ## or 1

        x_saliency =  self.x_saliency
        x_saliency =  x_saliency.clone().view(batch, -1)
        x_saliency = x_saliency.mean(1)  
        
        loss = res + x_saliency * 1.5
        loss = loss.mean(0) 
        return loss

    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        #--------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins + 1e-10)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed
            
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def model(args, pretrained=True):
    
    model = Model(args)
    if pretrained:
        pretrained_dict = torch.load('mobilenet_v1_with_relu_69_5.pth')
        model_dict = model.state_dict()
        model_conv_name = []

        for i, (k, v) in enumerate(model_dict.items()):
            if 'tracked' in k[-7:]:
                continue
            model_conv_name.append(k)
        for i, (k, v) in enumerate(pretrained_dict.items()):
            model_dict[model_conv_name[i]] = v 
        model.load_state_dict(model_dict)
    print("pretrained weight load complete..")
    return model