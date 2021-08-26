#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 07:49:53 2020

@author: aven
"""

import random

import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage import io,data
import albumentations as A
import os
import sys

data=sys.argv[1]

for file in os.listdir('./datasets/'+data+'/testA'):

    image = cv2.imread('./datasets/'+data+'/testA/'+file)
    print('./datasets/'+data+'/testA/'+file) 
    mask = cv2.imread('./datasets/'+data+'/testA_label/'+file, cv2.IMREAD_GRAYSCALE)
    for num in range(10):
        aug = A.ElasticTransform(p=1, alpha=600, sigma=600 * 0.05, alpha_affine=600 * 0.03)
        random.seed(num)
        augmented = aug(image=image, mask=mask)
        transformed_image = augmented['image']
        mask_elastic = augmented['mask']
        print('./datasets/'+data+'/testA/'+file[:-4]+str(num))
        cv2.imwrite('./'+data+'_paper_aug/img_aug/'+file[:-4]+'_'+str(num)+'.png',transformed_image)
        cv2.imwrite('./'+data+'_paper_aug/label/'+file[:-4]+'_'+str(num)+'.png',mask_elastic)
