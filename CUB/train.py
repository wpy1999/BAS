import os
import argparse
import torch
import torch.nn as nn 
from Model import *
from DataLoader import ImageDataset
from torch.autograd import Variable
from utils.accuracy import *
from utils.lr import *
from utils.optimizer import *
import os
import random
import time
seed = 6
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        ##  path
        self.parser.add_argument('--root', type=str, default='CUB_200_2011')
        self.parser.add_argument('--test_gt_path', type=str, default='CUB_200_2011/test_bounding_box.txt')
        self.parser.add_argument('--num_classes', type=int, default=200)
        self.parser.add_argument('--test_txt_path', type=str, default='CUB_200_2011/test_list.txt')
        ##  save
        self.parser.add_argument('--save_path', type=str, default='logs')
        self.parser.add_argument('--load_path', type=str, default='VGG.pth.tar')
        ##  dataloader
        self.parser.add_argument('--crop_size', type=int, default=224)
        self.parser.add_argument('--resize_size', type=int, default=256) 
        self.parser.add_argument('--num_workers', type=int, default=1)
        self.parser.add_argument('--nest', action='store_true')
        ##  train
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--epochs', type=int, default=100)
        self.parser.add_argument('--phase', type=str, default='train') ## train / test
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument('--weight_decay', type=float, default=5e-4)
        self.parser.add_argument('--power', type=float, default=0.9)
        self.parser.add_argument('--momentum', type=float, default=0.9)
        ##  model
        self.parser.add_argument('--arch', type=str, default='vgg')   ##  choose  [ vgg, resnet, inception, mobilenet ]         
        ##  show
        self.parser.add_argument('--show_step', type=int, default=94)
        ##  GPU'
        self.parser.add_argument('--gpu', type=str, default='0')
        
    def parse(self):
        opt = self.parser.parse_args()
        opt.arch = opt.arch     
        return opt

args = opts().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

lr = args.lr

if __name__ == '__main__':
    if args.phase == 'train':
        MyData = ImageDataset(args)
        MyDataLoader = torch.utils.data.DataLoader(dataset=MyData, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers)
        ##  model
        model = eval(args.arch).model(args, pretrained=True).cuda()
        model.train()
        ##  optimizer 
        optimizer = get_optimizer(model, args)
        loss_func = nn.CrossEntropyLoss().cuda()
        epoch_loss = 0

        print('Train begining!')
        for epoch in range(0, args.epochs):
            ##  accuracy
            cls_acc_1 = AverageMeter()
            cls_acc_2 = AverageMeter()
            loss_epoch_1 = AverageMeter()
            loss_epoch_2 = AverageMeter()
            loss_epoch_3 = AverageMeter()
            poly_lr_scheduler(optimizer, epoch, decay_epoch=80)
            for step, (path, imgs, label) in enumerate(MyDataLoader):                
                imgs, label = Variable(imgs).cuda(), label.cuda()
                ##  backward
                optimizer.zero_grad()
                output1,output2= model(imgs, label, 1)
                label = label.long()
                pred = torch.max(output1, 1)[1]  
                loss_1 = loss_func(output1, label).cuda()
                loss_2 = loss_func(output2, label).cuda()
                loss_3 = model.bas_loss().cuda() 
                if args.arch == 'vgg':
                    loss =  loss_1  + loss_2 * 0.0 + loss_3 
                else:
                    loss =  loss_1  + loss_2 * 0.5 + loss_3 
                loss.backward()
                optimizer.step() 
                ##  count_accuracy
                cur_batch = label.size(0)
                cur_cls_acc_1 = 100. * compute_cls_acc(output1, label) 
                cls_acc_1.updata(cur_cls_acc_1, cur_batch)
                cur_cls_acc_2 = 100. * compute_cls_acc(output2, label) 
                cls_acc_2.updata(cur_cls_acc_2, cur_batch)
                loss_epoch_1.updata(loss_1.data, 1)
                loss_epoch_2.updata(loss_2.data, 1)
                loss_epoch_3.updata(loss_3.data, 1)
                if (step+1) % args.show_step == 0 :
                    print('  Epoch:[{}/{}]\t step:[{}/{}]\tcls_loss_1:{:.3f}\tcls_loss_2:{:.3f}\tbas_loss:{:.3f}\t cls_acc_1:{:.2f}%\tcls_acc_2:{:.2f}%'.format(
                             epoch+1, args.epochs, step+1, len(MyDataLoader), loss_epoch_1.avg, loss_epoch_2.avg, loss_epoch_3.avg, cls_acc_1.avg, cls_acc_2.avg
                    ))
                    
            if epoch % 1 == 0:
                torch.save(model.state_dict(), os.path.join('logs/' + args.arch  +'/'+ args.arch +  str(epoch) +'.pth.tar'),_use_new_zipfile_serialization=False)

        






