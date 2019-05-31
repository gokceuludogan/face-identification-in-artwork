#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 19:45:48 2019

@author: gokce
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import local_binary_pattern

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def lbp_features(image_path):
    image_data = rgb2gray(image_path)
    width = image_data.shape[1]
    height = image_data.shape[0]
    image_data_padded = np.pad(image_data, pad_width=1, mode='constant', constant_values=0)
    image_lbp = np.zeros((height, width, 3), np.uint8)
    
    for x in range(1, height+1):
        for y in range(1, width+1):
            square = image_data_padded[x-1:x+2, y-1:y+2]
            center = image_data_padded[x, y]
            square_bin = square.copy()
            square_bin[square < center] = 0
            square_bin[square >= center] = 1
            square_bin[1, 1] = 0
            values = square_bin.flatten()
            power_val = np.array([1, 2, 4, 8, 0, 16, 32, 64, 128])
            lbp_value = np.dot(values, power_val)
            image_lbp[x-1, y-1] = lbp_value
    #plt.imshow(image_data, cmap = plt.get_cmap('gray'))        
    #plt.imshow(image_lbp, cmap = plt.get_cmap('gray'))        
    eps = 1e-7
    (hist, _) = np.histogram(image_lbp.ravel(), bins =np.arange(0,  60), range = (0, 254))
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist 

def lbp_skimage(img, num_points, radius, eps=1e-7):
    img = rgb2gray(img)
    lkp = local_binary_pattern(img, num_points, radius, 'uniform')     
    (hist,_) = np.histogram(lkp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist

#
#def full_lbp_features(image_path):
#    image = Image.open(image_path).convert("L")
#    width = image.size[0]
#    height = image.size[1]
#    image_data = np.array(image.getdata()).reshape(height, width)
#    image_data_padded = np.pad(image_data, pad_width=1, mode='constant', constant_values=0)
#    image_lbp = np.zeros((height, width, 3), np.uint8)
#    
#    for x in range(1, height+1):
#        for y in range(1, width+1):
#            square = image_data_padded[x-1:x+2, y-1:y+2]
#            center = image_data_padded[x, y]
#            square_bin = square.copy()
#            square_bin[square < center] = 0
#            square_bin[square >= center] = 1
#            square_bin[1, 1] = 0
#            values = square_bin.flatten()
#            power_val = np.array([1, 2, 4, 8, 0, 16, 32, 64, 128])
#            lbp_value = np.dot(values, power_val)
#            image_lbp[x-1, y-1] = lbp_value
#    #plt.imshow(image_data, cmap = plt.get_cmap('gray'))        
#    #plt.imshow(image_lbp, cmap = plt.get_cmap('gray'))        
#    eps = 1e-7
#    (hist, _) = np.histogram(image_lbp.ravel(), bins=np.arange(0, 256))
#    hist = hist.astype("float")
#    hist /= (hist.sum() + eps)
#    return hist     
#    #plt.plot(hist, color = "black")


#image_path = '/media/gokce/Data/BOUN/Spring19/Cmpe58Z/term-project/data/johannes_vermeer/00000009.jpg'
#lbp_features(image_path)        