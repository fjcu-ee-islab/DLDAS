# -*- coding:utf-8 -*-

from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob
import random
import math
import sys
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='display an integer')
parser.add_argument('--SupTL_alpha', type=str, help='display an integer')
parser.add_argument('--SupTL_beta', type=str, help='display an integer')
parser.add_argument('--SupTL_gamma', type=str, help='display an integer')
parser.add_argument('--SupTL_k', type=str, help='display an integer')
parser.add_argument('--g_aug', type=str, help='display an integer')
parser.add_argument('--result', type=str, help='display an integer')
parser.add_argument('--test_dir', type=str, help='display an integer')
parser.add_argument('--test_type', type=str, help='display an integer')
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
test_dir=args.test_dir

npy_dir='./npydata/'+data+'/'+test_type
result_dir='./results/'+data+'/'+result+'/'+augt+'/'+lossfunc+'_alpha_'+weight_alpha+'_beta_'+weight_beta+'_k_'+weight_k+'_gamma_'+weight_gamma+'/'+test_type+'_test'



if not os.path.exists(npy_dir):
      os.makedirs(npy_dir)

if not os.path.exists(result_dir):
      os.makedirs(result_dir)


class dataProcess(object):
    def __init__(self, out_rows, out_cols,test_path=test_dir, npy_path=npy_dir, img_type="png"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    def label2class(self, label):
        x = np.zeros([self.out_rows, self.out_cols, 3])
        for i in range(self.out_rows):
            for j in range(self.out_cols):
                x[i, j, int(label[i][j])] = 1  # 属于第m类，第三维m处值为1
        return x


    def create_test_data(self):
        i = 0
        print('Creating test images...')
        imgs = sorted(glob.glob(self.test_path + "/*." + self.img_type))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        testpathlist = []

        for imgname in imgs:
            testpath = imgname
            testpathlist.append(testpath)
            img = load_img(testpath, grayscale=False, target_size=[512, 512])
            img = img_to_array(img)
            imgdatas[i] = img
            print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1

        txtname = result_dir+'/pic.txt'
        with open(txtname, 'w') as f:
            for i in range(len(testpathlist)):
                f.writelines(testpathlist[i] + '\n')
        print('loading done')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')


    def load_test_data(self):
        print('-' * 30)
        print('load test1 images...')
        print('-' * 30)
        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
#        imgs_test /= 255
        return imgs_test
   



if __name__ == "__main__":
    mydata = dataProcess(512, 512)
    mydata.create_test_data()   
