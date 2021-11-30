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

    intersec[0] = max(j, bbox[0])
    intersec[1] = max(i, bbox[1])
    intersec[2] = min(j + w, bbox[2])
    intersec[3] = min(i + h, bbox[3])
    return intersec


def normalize_intersec(i, j, h, w, intersec):
    '''
    return: normalize into [0, 1]
    '''

    intersec[0] = (intersec[0] - j) / w
    intersec[2] = (intersec[2] - j) / w
    intersec[1] = (intersec[1] - i) / h
    intersec[3] = (intersec[3] - i) / h
    return intersec
