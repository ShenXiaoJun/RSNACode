# encoding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import cv2
import csv
import os
import string
import datetime

# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 训练数据的图像分辨率，这可以作为Example中的一个属性
pic_width = 1024
pic_height = 1024
pic_pixel = pic_width*pic_height
label_topx = [0,0,0,0]
label_topy = [0,0,0,0]
label_width =[0,0,0,0]
label_height=[0,0,0,0]
label_lowx = [0,0,0,0]
label_lowy = [0,0,0,0]
label_class =[0,0,0,0]

pic_dir         = "/home/shenxj/RSNA/code/test/pic_test"
class_info_file = "/home/shenxj/RSNA/code/test/stage_1_detailed_class_info.csv"
labels_file     = "/home/shenxj/RSNA/code/test/stage_1_train_labels.csv"
# 输出TFRcord文件的地址
tf_record_file  = "/home/shenxj/RSNA/code/test/dicom100.tfrecords"


starttime = datetime.datetime.now()
pic_index=0

with tf.python_io.TFRecordWriter(tf_record_file) as writer:

    pic_list = os.listdir(pic_dir) #列出文件夹下所有的目录与文件
    for i in range(0,len(pic_list)):
        pic_name = pic_list[i]
        pic_path = os.path.join(pic_dir,pic_name)
        pic_name_only = pic_name.split('.')[0]
        # print "-debug-"+"pic_name: "+pic_name
        # print "-debug-"+"pic_path: "+pic_path
        print ("-debug-pic_name_only: %s" % pic_name_only)
        if os.path.isfile(pic_path) == False:
            continue

        pic_index += 1

        # 获取image_raw,将图像矩阵转化成一个字符串
        image = cv2.imread(pic_path)
        image_raw = image.tostring()

        #获取label_class[]
        # label_class 0:Normal; 1:No Lung Opacity / Not Normal; 2:Lung Opacity
        label_class = [0, 0, 0, 0]
        with open(class_info_file, 'r') as class_info_csv:
            class_info_reader = csv.DictReader(class_info_csv)
            label_index = 0
            for row in class_info_reader:
                if pic_name_only in row.get('patientId'):
                    if label_index > 3:
                        print "In get label_class label_index > 3!"
                        continue
                    if row.get('class') == "Normal":
                        label_class[label_index]=0
                    elif row.get('class') == "No Lung Opacity / Not Normal":
                        label_class[label_index]=1
                    elif row.get('class') == "Lung Opacity":
                        label_class[label_index]=2
                    print("-debug-label_class[%d]:%d" % (label_index, label_class[label_index]))
                    label_index += 1

        #获取label_box
        label_topx = [0, 0, 0, 0]
        label_topy = [0, 0, 0, 0]
        label_lowx = [0, 0, 0, 0]
        label_lowy = [0, 0, 0, 0]
        with open(labels_file, 'r') as labels_csv:
            labels_reader = csv.DictReader(labels_csv)
            label_index = 0
            for row in labels_reader:
                if pic_name_only in row.get('patientId'):
                    if label_index > 3:
                        print "In get label_box label_index > 3!"
                        continue
                    if row.get('Target') == "0":
                        continue
                    label_topx[label_index] = string.atoi(row.get('x').split('.')[0])
                    label_topy[label_index] = string.atoi(row.get('y').split('.')[0])
                    label_lowx[label_index] = label_topx[label_index] + string.atoi(row.get('width').split('.')[0])
                    label_lowy[label_index] = label_topy[label_index] + string.atoi(row.get('height').split('.')[0])
                    print("-debug-label_box[%d]:[%d],[%d],[%d],[%d]" %
                          (label_index,
                           label_topx[label_index], label_topy[label_index],
                           label_lowx[label_index], label_lowy[label_index]))
                    label_index += 1

        # 将一个样例转化为Example Protocol Buffer,并将所有的信息写入这个数据结构。
        example = tf.train.Example(features=tf.train.Features(feature={
            'pic_pixel': _int64_feature(pic_pixel),

            'label_topx_0': _int64_feature(label_topx[0]),
            'label_topy_0': _int64_feature(label_topy[0]),
            'label_lowx_0': _int64_feature(label_lowx[0]),
            'label_lowy_0': _int64_feature(label_lowy[0]),
            'label_class_0': _int64_feature(label_class[0]),

            'label_topx_1': _int64_feature(label_topx[1]),
            'label_topy_1': _int64_feature(label_topy[1]),
            'label_lowx_1': _int64_feature(label_lowx[1]),
            'label_lowy_1': _int64_feature(label_lowy[1]),
            'label_class_1': _int64_feature(label_class[1]),

            'label_topx_2': _int64_feature(label_topx[2]),
            'label_topy_2': _int64_feature(label_topy[2]),
            'label_lowx_2': _int64_feature(label_lowx[2]),
            'label_lowy_2': _int64_feature(label_lowy[2]),
            'label_class_2': _int64_feature(label_class[2]),

            'label_topx_3': _int64_feature(label_topx[3]),
            'label_topy_3': _int64_feature(label_topy[3]),
            'label_lowx_3': _int64_feature(label_lowx[3]),
            'label_lowy_3': _int64_feature(label_lowy[3]),
            'label_class_3': _int64_feature(label_class[3]),

            'image_raw': _bytes_feature(image_raw)
        }))

        # 将一个Example写入TFRecord文件
        writer.write(example.SerializeToString())

endtime = datetime.datetime.now()
print ("-debug-use %d s"%(endtime - starttime).seconds)
print ("-debug-pic_index:%d"%pic_index)