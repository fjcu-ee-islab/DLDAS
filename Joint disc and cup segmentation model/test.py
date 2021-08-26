# -*- coding:utf-8 -*-
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import random as rn
rn.seed(1)
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import array_to_img
import keras
import cv2
#from data import dataProcess
import segmentation_models_Sup_TL as sm
import matplotlib.pyplot as plt
import sys
import os
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import argparse
import shutil
import albumentations as A
import numpy as np
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config= tf.ConfigProto(gpu_options=gpu_options))
os.environ['PYTHONHASHSEED'] = '0'
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#os.environ["CUDA_DEVICE_ORDER"] = "0" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 設定 Keras 使用的 Session
K.set_session(sess)
#os.environ['TF_DETERMINISTIC_OPS'] = '1'
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='display an integer')
parser.add_argument('--SupTL_alpha', type=str, help='display an integer')
parser.add_argument('--SupTL_beta', type=str, help='display an integer')
parser.add_argument('--SupTL_gamma', type=str, help='display an integer')
parser.add_argument('--SupTL_k', type=str, help='display an integer')
parser.add_argument('--g_aug', type=str, help='display an integer')
parser.add_argument('--result', type=str, help='display an integer')
parser.add_argument('--test_type', type=str, help='display an integer')
parser.add_argument('--gt_dir', type=str, help='display an integer')
parser.add_argument('--model_n', type=str, help='display an integer')
args = parser.parse_args()


data=args.data
result=args.result
augt=args.g_aug
lossfunc='Sup_TL'
weight_alpha=args.SupTL_alpha
weight_beta=args.SupTL_beta
weight_k=args.SupTL_k
weight_gamma=args.SupTL_gamma
test_type=args.test_type
gt_dir=args.gt_dir
model_name=args.model_n

npy_dir='./npydata/'+data+'/'+test_type
result_dir='./results/'+data+'/'+result+'/'+augt+'/'+lossfunc+'_alpha_'+weight_alpha+'_beta_'+weight_beta+'_k_'+weight_k+'_gamma_'+weight_gamma+'/'+test_type+'_test'



if not os.path.exists(npy_dir):
      os.mkdir(npy_dir)

if not os.path.exists(result_dir):
      os.makedirs(result_dir)

      
if not os.path.exists(result_dir+'/cuppic'):
      os.mkdir(result_dir+'/cuppic')
if not os.path.exists(result_dir+'/discpic'):
      os.mkdir(result_dir+'/discpic')

      
      
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

# classes for data loading and preprocessing
class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols

#    def load_data(self):
#        mydata = dataProcess(self.img_rows, self.img_cols)
#        imgs_test = mydata.load_test_data()      
#        return imgs_test

    def load_test_data(self):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(npy_dir + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
#        imgs_test /= 255
        return imgs_test

#    def get_unet(self):
#        w1=float(weight1)
#        w2=float(weight2)
#        model = sm.Unet("inceptionv3", classes=3,encoder_weights='imagenet',activation='softmax')
#        SupTL = sm.losses.SuppressedTverskyLoss(alpha=0.5 ,beta=0.5,k=w1,gamma=w2)
#        print(lossfunc+'_k_'+weight1+'_gamma_'+weight2)
#        model.compile(optimizer=Adam(lr=1e-4), loss=SupTL, metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5),"accuracy"])

        #return model
    def test(self):
        #K.set_learning_phase(1)
        model = sm.Unet("inceptionv3",classes=3,activation='softmax')
        #model = load_model('./model/'+model_name+'.h5',custom_objects={"suppressedtversky_loss": sm.losses.SuppressedTverskyLoss})
        model.load_weights('./model/'+model_name+'.hdf5')
        #model.load_weights('./model/'+model_name+'.h5')
        imgs_test= self.load_test_data()
        print('predict test data') 
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save(result_dir+'/imgs_mask_test_'+test_type+'.npy', imgs_mask_test)

 


    def save_img(self):
        print("array to image")
        imgs = np.load(result_dir+'/imgs_mask_test_'+test_type+'.npy')
        dnum=imgs.shape[0]
        piclist = []
        for line in open(result_dir+"/pic.txt"):
            line = line.strip()
            picname = line.split('/')[-1]
            piclist.append(picname)
        sumdice_disk=0.0
        sumiou_disk=0.0
        sumdice_cup=0.0
        sumiou_cup=0.0
        for i in range(imgs.shape[0]):
            print('{:d}/{:d}'.format(i,imgs.shape[0]))
            path_disk = result_dir+'/discpic/'+piclist[i]
            path_cup = result_dir+'/cuppic/'+piclist[i]
            pathori=result_dir+'/'+piclist[i]
            img = np.zeros((imgs.shape[1], imgs.shape[2], 3), dtype=np.uint8)
            for k in range(len(img)):
                for j in range(len(img[k])):  # cv2.imwrite也是BGR顺序
                    num = np.argmax(imgs[i][k][j])
                    if num == 0:
                        img[k][j] = [0, 0, 0]
                    elif num == 1:
                        img[k][j] = [128, 128, 128]
                    elif num == 2:
                        img[k][j] = [255, 255, 255]
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            cv_pic=cv2.cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            binary, cv_disk = cv2.threshold(cv_pic, 127, 255, cv2.THRESH_BINARY)
            binary, cv_cup = cv2.threshold(cv_pic, 254, 255, cv2.THRESH_BINARY)
            if 'jsrt' in data:
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
            f=open(result_dir+'/dice_score_class1_'+data+'_'+test_type+'.txt','a')
            f.write(piclist[i]+'\t'+'dice_score:'+'\t')
            f.write(wdice_disk)
            f.write('\n')
            f=open(result_dir+'/iou_class1_'+data+'_'+test_type+'.txt','a')
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
            f=open(result_dir+'/dice_score_class2_'+data+'_'+test_type+'.txt','a')
            f.write(piclist[i]+'\t'+'dice_score:'+'\t')
            f.write(wdice_cup)
            f.write('\n')
            f=open(result_dir+'/iou_class2_'+data+'_'+test_type+'.txt','a')
            f.write(piclist[i]+'\t'+'iou:'+'\t')
            f.write(wiou_cup)
            f.write('\n')
        #disk
        meaniou_disk=sumiou_disk/dnum
        meandice_disk=sumdice_disk/dnum
        wmeandice_disk=str(meandice_disk)
        f=open(result_dir+'/dice_score_class1_'+data+'_'+test_type+'.txt','a')
        f.write('\t'+'meandice_score:'+'\n')
        f.write(wmeandice_disk)
        f.close()
        wmeaniou_disk=str(meaniou_disk)
        f=open(result_dir+'/iou_class1_'+data+'_'+test_type+'.txt','a')
        f.write('\t'+'meaniou:'+'\n')
        f.write(wmeaniou_disk)
        f.close()
        #cup
        meaniou_cup=sumiou_cup/dnum
        meandice_cup=sumdice_cup/dnum
        wmeandice_cup=str(meandice_cup)
        f=open(result_dir+'/dice_score_class2_'+data+'_'+test_type+'.txt','a')
        f.write('\t'+'meandice_score:'+'\n')
        f.write(wmeandice_cup)
        f.close()
        wmeaniou_cup=str(meaniou_cup)
        f=open(result_dir+'/iou_class2_'+data+'_'+test_type+'.txt','a')
        f.write('\t'+'meaniou:'+'\n')
        f.write(wmeaniou_cup)
        f.close()



if __name__ == '__main__':
    myunet = myUnet()
    myunet.test()
    myunet.save_img()    
    
