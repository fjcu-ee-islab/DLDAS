#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 23:01:30 2020

@author: aven
"""

import numpy as np
import glob
import sys
import os
import math
import cv2
import shutil

trainnum=sys.argv[1]
data=sys.argv[2]
style=sys.argv[3]

data_dir='./'+data+'_paper_synthesis/'+trainnum
data_save_dir1='./'+data+'_paper_aug'
data_save_dir='./'+data+'_paper_aug/img_aug'
label_save_dir='./'+data+'_paper_aug/label'

if not os.path.exists(data_dir):
      os.mkdir(data_dir)
if not os.path.exists(data_save_dir1):
      os.mkdir(data_save_dir1)
if not os.path.exists(data_save_dir):
      os.mkdir(data_save_dir)
if not os.path.exists(label_save_dir):
      os.mkdir(label_save_dir)

def renameFile(fileDir,targetDir,style):
    # 1
    files = os.listdir(fileDir)
    for name in files:     
        for i in range(int(style)):
            print(fileDir+'/'+name)
            img=cv2.imread(fileDir+'/'+name)
            print(fileDir+'/'+name[:-4]+'_s'+str(i)+".png")
            cv2.imwrite(targetDir+'/'+name[:-4]+'_s'+str(i)+".png",img)

def copyFile(fileDir,labelDir,labelDir2):
    # 1
    files = os.listdir(fileDir)
    for name in files:     
        print(fileDir+'/'+name)
        shutil.copy(labelDir+'/'+name, labelDir2+'/'+name)

i=0
for x in range(int(style)):
    if x>=10:
        files= os.listdir(data_dir+'_'+str(i))
    else:
        files= os.listdir(data_dir+'_0'+str(i))
    for file in files:
        print(x)
        if x>=10:
           img=cv2.imread(data_dir+'_'+str(i)+'/'+file)
           print(data_dir+'_'+str(i)+'/'+file) 
        else:   
           img=cv2.imread(data_dir+'_0'+str(i)+'/'+file)
           print(data_dir+'_0'+str(i)+'/'+file)
#    print(data_save_dir+'/'+file[:-4]+'_s'+str(i)+".png")
        cv2.imwrite(data_save_dir+'/'+file[:-4]+'_s'+str(i)+".png",img)
    i=i+1

copyFile('./datasets/'+data+'/testA','./datasets/'+data+'/testA_label','./'+data+'_paper_aug/label')
copyFile('./datasets/'+data+'/testA','./datasets/'+data+'/testA','./'+data+'_paper_aug/img_aug')
renameFile('./datasets/'+data+'/testA_label','./'+data+'_paper_aug/label',9)
