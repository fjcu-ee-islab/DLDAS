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
parser.add_argument('--train_dir', type=str, help='display an integer')
parser.add_argument('--val_dir', type=str, help='display an integer')
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

train_data_dir=args.train_dir+'/image'
train_label_dir=args.train_dir+'/label'

val_data_dir=args.val_dir+'/image'
val_label_dir=args.val_dir+'/label'

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

class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['disc', 'cup']
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        image=cv2.resize(image,(512,512))
        mask=cv2.resize(mask,(512,512))        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        image=img_to_array(image)
        mask=img_to_array(mask)
        imgge = image.astype('float32')
        mask = mask.astype('float32')
#        image /= 255
#        mask /= 255             
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
    
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[self.indexes[j]])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
#########################################################################Augmentation
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    if augt=='Contrast+Gamma9':
    	train_transform = [A.HorizontalFlip(p=0.5),A.RandomContrast(limit=0.9,p=0.9),A.RandomGamma(gamma_limit=(10,190),p=0.9),A.Lambda(mask=round_clip_0_1)]
    elif augt=='Contrast+Gamma8':
    	train_transform = [A.HorizontalFlip(p=0.5),A.RandomContrast(limit=0.8,p=0.9),A.RandomGamma(gamma_limit=(20,180),p=0.9),A.Lambda(mask=round_clip_0_1)]
    elif augt=='Contrast+Gamma7':
    	train_transform = [A.HorizontalFlip(p=0.5),A.RandomContrast(limit=0.7,p=0.9),A.RandomGamma(gamma_limit=(30,170),p=0.9),A.Lambda(mask=round_clip_0_1)]
    elif augt=='Contrast+Gamma6':
    	train_transform = [A.HorizontalFlip(p=0.5),A.RandomContrast(limit=0.6,p=0.9),A.RandomGamma(gamma_limit=(40,160),p=0.9),A.Lambda(mask=round_clip_0_1)]
    elif augt=='Contrast+Gamma5':
    	train_transform = [A.HorizontalFlip(p=0.5),A.RandomContrast(limit=0.5,p=0.9),A.RandomGamma(gamma_limit=(50,150),p=0.9),A.Lambda(mask=round_clip_0_1)]
    elif augt=='Contrast+Gamma4':
    	train_transform = [A.HorizontalFlip(p=0.5),A.RandomContrast(limit=0.4,p=0.9),A.RandomGamma(gamma_limit=(60,140),p=0.9),A.Lambda(mask=round_clip_0_1)]
    elif augt=='Contrast+Gamma3':
    	train_transform = [A.HorizontalFlip(p=0.5),A.RandomContrast(limit=0.3,p=0.9),A.RandomGamma(gamma_limit=(70,130),p=0.9),A.Lambda(mask=round_clip_0_1)]
    elif augt=='Contrast+Gamma2':
    	train_transform = [A.HorizontalFlip(p=0.5),A.RandomContrast(limit=0.2,p=0.9),A.RandomGamma(gamma_limit=(80,120),p=0.9),A.Lambda(mask=round_clip_0_1)]
    elif augt=='Contrast+Gamma1':
    	train_transform = [A.HorizontalFlip(p=0.5),A.RandomContrast(limit=0.1,p=0.9),A.RandomGamma(gamma_limit=(90,110),p=0.9),A.Lambda(mask=round_clip_0_1)]
    elif augt=='Contrast+Gamma2+crop':
    	train_transform = [A.RandomCrop(height=512, width=512,p=0.5),A.HorizontalFlip(p=0.5),A.RandomContrast(limit=0.2,p=0.9),A.RandomGamma(gamma_limit=(80,120),p=0.9),A.Lambda(mask=round_clip_0_1)]
    elif augt=='Contrast+Gamma1+crop':
    	train_transform = [A.RandomCrop(height=512, width=512,p=0.5),A.HorizontalFlip(p=0.5),A.RandomContrast(limit=0.1,p=0.9),A.RandomGamma(gamma_limit=(90,110),p=0.9),A.Lambda(mask=round_clip_0_1)]
    elif augt=='Contrast+Gamma1+elastic':
    	train_transform = [A.ElasticTransform(p=0.5, alpha=600, sigma=600 * 0.05, alpha_affine=600 * 0.03),A.HorizontalFlip(p=0.5),A.RandomContrast(limit=0.1,p=0.9),A.RandomGamma(gamma_limit=(90,110),p=0.9),A.Lambda(mask=round_clip_0_1)]
    elif augt=='Contrast+Gamma2+elastic':
    	train_transform = [A.ElasticTransform(p=0.5, alpha=600, sigma=600 * 0.05, alpha_affine=600 * 0.03),A.HorizontalFlip(p=0.5),A.RandomContrast(limit=0.2,p=0.9),A.RandomGamma(gamma_limit=(80,120),p=0.9),A.Lambda(mask=round_clip_0_1)]
    elif augt=='Contrast+Gamma2+elastic+crop':
    	train_transform = [A.ElasticTransform(p=0.5, alpha=600, sigma=600 * 0.05, alpha_affine=600 * 0.03),A.HorizontalFlip(p=0.5),A.RandomCrop(height=512, width=512,p=0.5),A.RandomContrast(limit=0.2,p=0.9),A.RandomGamma(gamma_limit=(80,120),p=0.9),A.Lambda(mask=round_clip_0_1)]
    elif augt=='None':
    	train_transform = []
    print(augt)


    
    return A.Compose(train_transform)
##########################################################################
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

    def get_unet(self):
        w_alpha=float(weight_alpha)
        w_beta=float(weight_beta)
        w_k=float(weight_k)
        w_gamma=float(weight_gamma)

        model = sm.Unet("inceptionv3", classes=3,encoder_weights='imagenet',activation='softmax')
        SupTL = sm.losses.SuppressedTverskyLoss(alpha=w_alpha ,beta=w_beta,k=w_k,gamma=w_gamma)
        print(lossfunc+'_k_'+weight_k+'_gamma_'+weight_gamma)
        model.compile(optimizer=Adam(lr=1e-4), loss=SupTL, metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5),"accuracy"])

        return model
    def test(self):
        K.set_learning_phase(1)
        CLASSES = ['disc', 'cup']
#        print("moving data")
        print("loading data")

        imgs_test= self.load_test_data()

        train_dataset = Dataset(
        train_data_dir, 
        train_label_dir, 
        classes=CLASSES,
        augmentation=get_training_augmentation(), 
        )

        # Dataset for validation images
        valid_dataset = Dataset(
        val_data_dir, 
        val_label_dir, 
        classes=CLASSES, 
        )

        train_dataloader = Dataloder(train_dataset, batch_size=2, shuffle=True)
        valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)
        print("loading data done")
        model = self.get_unet()
        print("got unet")
        model_checkpoint = ModelCheckpoint('./model/'+model_name+'.hdf5', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True)
        #model_checkpoint = ModelCheckpoint('./model/'+model_name+'.h5', monitor='val_loss', verbose=1, save_best_only=True)
        reducelearning=ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=5, verbose=1)
        earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        print('Fitting model...')
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
            f=open(result_dir+'/dice_score_disk_'+data+'_'+test_type+'.txt','a')
            f.write(piclist[i]+'\t'+'dice_score:'+'\t')
            f.write(wdice_disk)
            f.write('\n')
            f=open(result_dir+'/iou_disk_'+data+'_'+test_type+'.txt','a')
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
            f=open(result_dir+'/dice_score_cup_'+data+'_'+test_type+'.txt','a')
            f.write(piclist[i]+'\t'+'dice_score_cup:'+'\t')
            f.write(wdice_cup)
            f.write('\n')
            f=open(result_dir+'/iou_cup_'+data+'_'+test_type+'.txt','a')
            f.write(piclist[i]+'\t'+'iou:'+'\t')
            f.write(wiou_cup)
            f.write('\n')
        #disk
        meaniou_disk=sumiou_disk/dnum
        meandice_disk=sumdice_disk/dnum
        wmeandice_disk=str(meandice_disk)
        f=open(result_dir+'/dice_score_disk_'+data+'_'+test_type+'.txt','a')
        f.write('\t'+'meandice_score:'+'\n')
        f.write(wmeandice_disk)
        f.close()
        wmeaniou_disk=str(meaniou_disk)
        f=open(result_dir+'/iou_disk_'+data+'_'+test_type+'.txt','a')
        f.write('\t'+'meaniou:'+'\n')
        f.write(wmeaniou_disk)
        f.close()
        #cup
        meaniou_cup=sumiou_cup/dnum
        meandice_cup=sumdice_cup/dnum
        wmeandice_cup=str(meandice_cup)
        f=open(result_dir+'/dice_score_cup_'+data+'_'+test_type+'.txt','a')
        f.write('\t'+'meandice_score:'+'\n')
        f.write(wmeandice_cup)
        f.close()
        wmeaniou_cup=str(meaniou_cup)
        f=open(result_dir+'/iou_cup_'+data+'_'+test_type+'.txt','a')
        f.write('\t'+'meaniou:'+'\n')
        f.write(wmeaniou_cup)
        f.close()



if __name__ == '__main__':
    myunet = myUnet()
    myunet.test()
    myunet.save_img()    
    
