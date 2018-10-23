# encoding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cv2
import csv
import os
import string
import datetime
import shutil


# 训练数据的图像分辨率，这可以作为Example中的一个属性

pic_dir         = "/home/shenxj/RSNA/data/stage_1_train_jpg"
normal_dir      = "/home/shenxj/RSNA/code/test/pic_normal/"
not_normal_dir      = "/home/shenxj/RSNA/code/test/pic_not_normal/"
#class_info_file = "/home/shenxj/RSNA/code/test/stage_1_detailed_class_info.csv"
labels_file     = "/home/shenxj/RSNA/code/test/stage_1_train_labels.csv"


starttime = datetime.datetime.now()
pic_index=0

pic_list = os.listdir(pic_dir) #列出文件夹下所有的目录与文件
for i in range(0,len(pic_list)):
    pic_name = pic_list[i]
    pic_path = os.path.join(pic_dir,pic_name)
    pic_name_only = pic_name.split('.')[0]
    print ("-debug-pic_name_only: %s" % pic_name_only)
    if os.path.isfile(pic_path) == False:
        continue



    #获取target
    target = 0
    with open(labels_file, 'r') as labels_csv:
        labels_reader = csv.DictReader(labels_csv)
        for row in labels_reader:
            if pic_name_only in row.get('patientId'):
                if row.get('Target') == "0":
                    target = 0
                else:
                    target = 1

    #复制图片到对应文件夹
    if target == 0:
        shutil.copyfile(pic_path,normal_dir+pic_name_only+".jpg")
    elif target == 1:
        shutil.copyfile(pic_path, not_normal_dir + pic_name_only + ".jpg")

    pic_index += 1
    if pic_index >= 100:
        break;

endtime = datetime.datetime.now()
print ("-debug-use %d s"%(endtime - starttime).seconds)
print ("-debug-pic_index:%d"%pic_index)