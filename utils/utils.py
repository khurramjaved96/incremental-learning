import numpy as np
import torch.utils.data as td
from PIL import Image


def resizeImage(img,factor):
    '''
    
    :param img: 
    :param factor: 
    :return: 
    '''
    img2 = np.zeros(np.array(img.shape)*factor)

    for a in range(0,img.shape[0]):
        for b in range(0,img.shape[1]):
            img2[a*factor:(a+1)*factor, b*factor:(b+1)*factor] = img[a,b]
    return img2