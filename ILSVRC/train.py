import os
import argparse
import torch
import torch.nn as nn 
from Model import *
from DataLoader import ImageDataset
from torch.autograd import Variable
from utils.loss import Loss
from utils.accuracy import *
from utils.optimizer import *
from utils.lr import *
import os
import random
import time
seed = 2

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
 
class opts(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        ##  path
        parser.add_argument('--root', type=str, default='/media/data/imagenet-1k')
        parser.add_argument('--test_gt_path', type=str, default='val_gt.txt')
        parser.add_argument('--num_classes', type=int, default=1000)
        parser.add_argument('--test_txt_path', type=str, default='val_list.txt')

        parser.add_argument('--save_path', type=str, default='logs')
        parser.add_argument('--load_path', type=str, default='VGG.pth.tar')
        ##  image
        parser.add_argument('--crop_size', type=int, default=224)
        parser.add_argument('--resize_size', type=int, default=256) 
        ##  dataloader
        parser.add_argument('--num_workers', type=int, default=8)
        parser.add_argument('--nest', action='store_true')
        ##  train
        parser.add_argument('--batch_size', type=int, default=32*4)
        parser.add_argument('--epochs', type=int, default=9)
        parser.add_argument('--pretrain', type=str, default='True')
        parser.add_argument('--phase', type=str, default='train') 
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--power', type=float, default=0.9)
        parser.add_argument('--momentum', type=float, default=0.9)
        ##  model
        self.parser.add_argument('--arch', type=str, default='vgg')  ## choosen  [ vgg, resnet, inception, mobilenet] 
        ##  show
        parser.add_argument('--show_step', type=int, default=500)
        ##  GPU'
        parser.add_argument('--gpu', type=str, default='0,1,2,3 ')

    def parse(self):
        opt = self.parser.parse_args()
        opt.arch = opt.arch     
        return opt

args = opts().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
lr = args.lr

if __name__ == '__main__':
    '''print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)'''
    print(torch.cuda.get_device_name(0))
    if args.phase == 'train':
        ##  data.
        MyData = ImageDataset(args)
        MyDataLoader = torch.utils.data.DataLoader(dataset=MyData, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers,pin_memory=True)
        ##  model
        model = eval(args.arch).model(args, pretrained=True)
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
        model.cuda(device=0)
        model = model
        model.train()
        ##  optimizer 
        optimizer = get_optimizer(model, args)
        
        print('Train begining!')
        for epoch in range(0, args.epochs):
            ##  accuracy
            cls_acc_1 = AverageMeter()
            loss_epoch_1 = AverageMeter()
            loss_epoch_2 = AverageMeter()
            poly_lr_scheduler(optimizer, epoch, decay_epoch=3)
            torch.cuda.synchronize()
            start = time.time()

            for step, (path, imgs, label) in enumerate(MyDataLoader):
                
                imgs, label = Variable(imgs).cuda(device=0), label.cuda(device=0)
                ##  backward
                optimizer.zero_grad()
                output1, loss_cls, loss_loc= model(imgs, label, N=1)
                loss_cls, loss_loc = loss_cls.mean(0), loss_loc.mean(0)
                label = label.long()
                loss = loss_cls + loss_loc 
                loss.backward()
                optimizer.step() 

                ##  count_accuracy
                cur_batch = label.size(0)
                cur_cls_acc_1 = 100. * compute_cls_acc(output1, label) 
                cls_acc_1.updata(cur_cls_acc_1, cur_batch)
                loss_epoch_1.updata(loss_cls.data, 1)
                loss_epoch_2.updata(loss_loc.data, 1)
                
                if (step+1) % args.show_step == 0 :
                    print('Epoch:[{}/{}]\tstep:[{}/{}]   loss_cls:{:.3f}   loss_bas:{:.3f}   epoch_acc_1:{:.2f}% '.format(
                            epoch+1, args.epochs, step+1, len(MyDataLoader), loss_epoch_1.avg,loss_epoch_2.avg, cls_acc_1.avg 
                    ))
            torch.save(model.state_dict(), os.path.join(args.save_path, args.arch + str(epoch)+'_'+ 'final' +'.pth.tar') ) 






