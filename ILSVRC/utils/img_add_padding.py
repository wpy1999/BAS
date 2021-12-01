import torch
import copy
import numpy 

def add_img_padding(img,left_padding, up_padding, right_padding, down_padding):  ## 左上和右下至少一方为0
      ##  img_size = [224, 224, 3]
    img_size = img.shape()
    img_new = np.zeros(img_size)
    flag_1 = True if (left_padding!=0 or up_padding!=0) else False
    flag_2 = True if (right_padding!=0 or down_padding!=0) else False
    if flag_1:
        if left_padding!=0 and up_padding!=0:
            img_new[up_padding:,left_padding:,:] = img[:-up_padding,:-left_padding,:]
        elif left_padding == 0 :
            img_new[up_padding:,:,:] = img[:-up_padding,:,:]
        else:
            img_new[:,left_padding:,:] = img[:,:-left_padding,:]
    elif flag_2:
            
    else:
        img_new = copy.deepcopy(img)