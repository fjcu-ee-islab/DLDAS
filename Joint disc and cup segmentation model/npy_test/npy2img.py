from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob
import random
import math
import sys
import os
import argparse
import cv2
from PIL import Image  
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--npy_name', type=str, help='display an integer')
parser.add_argument('--npy_dir', type=str, help='display an integer')
parser.add_argument('--gt_dir', type=str, help='display an integer')
parser.add_argument('--result_dir', type=str, help='display an integer')
parser.add_argument('--test_name', type=str, help='display an integer')
args = parser.parse_args()

result_dir=args.result_dir
npy_dir=args.npy_dir
npy_name=args.npy_name
test_name=args.test_name
gt_dir=args.gt_dir


if not os.path.exists(npy_dir):
      os.makedirs(npy_dir)

if not os.path.exists(result_dir):
      os.makedirs(result_dir)

      
if not os.path.exists(result_dir+'/class2_pic'):
      os.mkdir(result_dir+'/class2_pic')
if not os.path.exists(result_dir+'/class1_pic'):
      os.mkdir(result_dir+'/class1_pic')

      
      
if os.path.exists(result_dir+'/dice_score_cup.txt'):
      os.remove(result_dir+'/dice_score_cup.txt')
      os.remove(result_dir+'/iou_cup.txt')
      os.remove(result_dir+'/dice_score_disk.txt')
      os.remove(result_dir+'/iou_disk.txt')

def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")    
    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum

print("array to image")
imgs = np.load(npy_dir+'/'+npy_name+'.npy')
dnum=imgs.shape[0]
piclist = []
for line in open(npy_dir+'/'+npy_name+'_pic.txt'):
    line = line.strip()
    picname = line.split('/')[-1]
    piclist.append(picname)
sumdice_disk=0.0
sumiou_disk=0.0
sumdice_cup=0.0
sumiou_cup=0.0
for i in range(imgs.shape[0]):
    print('{:d}/{:d}'.format(i,imgs.shape[0]))
    path_disk = result_dir+'/class1_pic/'+piclist[i]
    path_cup = result_dir+'/class2_pic/'+piclist[i]
    pathori=result_dir+'/'+piclist[i]
    img = imgs[i,:,:,:]
    #print(img.shape)
    #img=Image.fromarray(img)
    cv_pic = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    #cv_pic=cv2.cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    binary, cv_disk = cv2.threshold(cv_pic, 127, 255, cv2.THRESH_BINARY)
    binary, cv_cup = cv2.threshold(cv_pic, 254, 255, cv2.THRESH_BINARY)
    if 'jsrt' in test_name:
        cv_disk=cv_disk-cv_cup 
    target_disk=cv2.imread(gt_dir+'/class1_answer/'+piclist[i],cv2.IMREAD_GRAYSCALE)
    target_disk= cv2.resize(target_disk, (512, 512), interpolation=cv2.INTER_CUBIC)
    target_cup=cv2.imread(gt_dir+'/class2_answer/'+piclist[i],cv2.IMREAD_GRAYSCALE)
    target_cup= cv2.resize(target_cup, (512, 512), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(pathori, img)
    cv2.imwrite(path_disk, cv_disk)
    cv2.imwrite(path_cup, cv_cup)
    #disk
    intersection = np.logical_and(target_disk, cv_disk)
    union = np.logical_or(target_disk, cv_disk)
    iou_score_disk = np.sum(intersection) / np.sum(union)
    sumiou_disk=sumiou_disk+iou_score_disk
    wiou_disk=str(iou_score_disk)
    dice_score_disk=dice(cv_disk, target_disk)
    sumdice_disk=sumdice_disk+dice_score_disk
    wdice_disk=str(dice_score_disk)
    f=open(result_dir+'/dice_score_class1_'+test_name+'.txt','a')
    f.write(piclist[i]+'\t'+'dice_score:'+'\t')
    f.write(wdice_disk)
    f.write('\n')
    f=open(result_dir+'/iou_class1_'+test_name+'.txt','a')
    f.write(piclist[i]+'\t'+'iou:'+'\t')
    f.write(wiou_disk)
    f.write('\n')
    #cup
    intersection = np.logical_and(target_cup, cv_cup)
    union = np.logical_or(target_cup, cv_cup)
    iou_score_cup = np.sum(intersection) / np.sum(union)
    sumiou_cup=sumiou_cup+iou_score_cup
    wiou_cup=str(iou_score_cup)
    dice_score_cup=dice(cv_cup, target_cup)
    sumdice_cup=sumdice_cup+dice_score_cup
    wdice_cup=str(dice_score_cup)
    f=open(result_dir+'/dice_score_class2_'+test_name+'.txt','a')
    f.write(piclist[i]+'\t'+'dice_score:'+'\t')
    f.write(wdice_cup)
    f.write('\n')
    f=open(result_dir+'/iou_class2_'+test_name+'.txt','a')
    f.write(piclist[i]+'\t'+'iou:'+'\t')
    f.write(wiou_cup)
    f.write('\n')
#disk
meaniou_disk=sumiou_disk/dnum
meandice_disk=sumdice_disk/dnum
wmeandice_disk=str(meandice_disk)
f=open(result_dir+'/dice_score_class1_'+test_name+'.txt','a')
f.write('\t'+'meandice_score:'+'\n')
f.write(wmeandice_disk)
f.close()
wmeaniou_disk=str(meaniou_disk)
f=open(result_dir+'/iou_class1_'+test_name+'.txt','a')
f.write('\t'+'meaniou:'+'\n')
f.write(wmeaniou_disk)
f.close()
#cup
meaniou_cup=sumiou_cup/dnum
meandice_cup=sumdice_cup/dnum
wmeandice_cup=str(meandice_cup)
f=open(result_dir+'/dice_score_class2_'+test_name+'.txt','a')
f.write('\t'+'meandice_score:'+'\n')
f.write(wmeandice_cup)
f.close()
wmeaniou_cup=str(meaniou_cup)
f=open(result_dir+'/iou_class2_'+test_name+'.txt','a')
f.write('\t'+'meaniou:'+'\n')
f.write(wmeaniou_cup)
f.close()
