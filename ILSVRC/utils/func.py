import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
def count_max(x):
    count_dict = {}
    for xlist in x:
        for item in xlist:
            if item==0:
                continue
            if item not in count_dict.keys():
                count_dict[item] = 0
            count_dict[item] += 1
    if count_dict == {}:
        return -1
    count_dict = sorted(count_dict.items(), key=lambda d:d[1], reverse=True)
    return count_dict[0][0]
    
def compute_intersec(i, j, h, w, bbox):
    '''
    intersection box between croped box and GT BBox
    '''
    intersec = copy.deepcopy(bbox)
    box_num = len(bbox) // 4
    for x in range(box_num):
        intersec[0+4*x] = max(j, bbox[0+4*x])
        intersec[1+4*x] = max(i, bbox[1+4*x])
        intersec[2+4*x] = min(j + w, bbox[2+4*x])
        intersec[3+4*x] = min(i + h, bbox[3+4*x])
    return intersec


def normalize_intersec(i, j, h, w, intersec):
    '''
    return: normalize into [0, 1]
    '''
    box_num = len(intersec) // 4
    for x in range(box_num):
        intersec[0+4*x] = (intersec[0+4*x] - j) / w
        intersec[2+4*x] = (intersec[2+4*x] - j) / w
        intersec[1+4*x] = (intersec[1+4*x] - i) / h
        intersec[3+4*x] = (intersec[3+4*x] - i) / h
    return intersec


