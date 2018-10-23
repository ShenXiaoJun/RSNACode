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

pic_dir         = "/home/shenxj/RSNA/code/test/pic_test"
#class_info_file = "/home/shenxj/RSNA/code/test/stage_1_detailed_class_info.csv"
labels_file     = "/home/shenxj/RSNA/code/test/stage_1_train_labels.csv"
# 输出TFRcord文件的地址
tf_record_file  = "/home/shenxj/RSNA/code/test/dicom100_nolabel_test.tfrecords"


starttime = datetime.datetime.now()
pic_index=0

with tf.python_io.TFRecordWriter(tf_record_file) as writer:

    pic_list = os.listdir(pic_dir) #列出文件夹下所有的目录与文件
    for i in range(0,len(pic_list)):
        pic_name = pic_list[i]
        pic_path = os.path.join(pic_dir,pic_name)
        pic_name_only = pic_name.split('.')[0]
        print ("-debug-pic_name_only: %s" % pic_name_only)
        if os.path.isfile(pic_path) == False:
            continue

        pic_index += 1

        # 获取image_raw,将图像矩阵转化成一个字符串
        image = cv2.imread(pic_path)
        image_raw = image.tostring()

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

        # 将一个样例转化为Example Protocol Buffer,并将所有的信息写入这个数据结构。
        example = tf.train.Example(features=tf.train.Features(feature={
            'target': _int64_feature(target),
            'image_raw': _bytes_feature(image_raw)
        }))

        # 将一个Example写入TFRecord文件
        writer.write(example.SerializeToString())

endtime = datetime.datetime.now()
print ("-debug-use %d s"%(endtime - starttime).seconds)
print ("-debug-pic_index:%d"%pic_index)