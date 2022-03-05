import torch
def get_optimizer(model, args):
        lr =args.lr        
        weight_list = [] 
        bias_list = []
        last_weight_list = []
        last_bias_list =[]
        for name, value in model.named_parameters():
            if 'copy' in name : 
                continue
            if 'classifier' in name : 
                if 'weight' in name:
                    last_weight_list.append(value)
                elif 'bias' in name:
                    last_bias_list.append(value)
            else:
                if 'weight' in name:
                    weight_list.append(value)
                elif 'bias' in name:
                    bias_list.append(value)
        optmizer = torch.optim.SGD([{'params': weight_list,
                                     'lr': lr},
                                    {'params': bias_list,
                                     'lr': lr*2},
                                    {'params': last_weight_list,
                                     'lr': lr*10},
                                    {'params': last_bias_list,
                                     'lr': lr*20}], momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        return optmizer