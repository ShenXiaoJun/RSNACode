# -*- coding:utf-8 -*-
#author: shenxj

import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
import tensorflow as tf
import cv2
import numpy as np
import csv
import string

print(__doc__)


filename='00436515-870c-4b36-a041-de91049b9ab4'
dcmPath='/home/shenxj/RSNA/data/RSNA/stage_1_train_images/'+filename+'.dcm'
jpgPath='/home/shenxj/RSNA/data/stage_1_train_jpg/'+filename+'.jpg'
labelPath='/home/shenxj/RSNA/data/RSNA/stage_1_train_labels.csv'
dataset = pydicom.dcmread(dcmPath)

# Normal mode:
print()
print("Filename.........:", filename)
print("Storage type.....:", dataset.SOPClassUID)
print()

pat_name = dataset.PatientName
display_name = pat_name.family_name + ", " + pat_name.given_name
print("Patient's name...:", display_name)
print("Patient id.......:", dataset.PatientID)
print("Modality.........:", dataset.Modality)
print("Study Date.......:", dataset.StudyDate)

if 'PixelData' in dataset:
    rows = int(dataset.Rows)
    cols = int(dataset.Columns)
    print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
        rows=rows, cols=cols, size=len(dataset.PixelData)))
    if 'PixelSpacing' in dataset:
        print("Pixel spacing....:", dataset.PixelSpacing)

# use .get() if not sure the item exists, and want a default value if missing
print("Slice location...:", dataset.get('SliceLocation', "(missing)"))

def convert_file(dcm_file_path, jpg_file_path):
    dicom_img = pydicom.read_file(dcm_file_path)
    img = dicom_img.pixel_array
    scaled_img = cv2.convertScaleAbs(img-np.min(img), alpha=(255.0 / min(np.max(img)-np.min(img), 10000)))
    cv2.imwrite(jpg_file_path, scaled_img)

convert_file(dcmPath, jpgPath)

with tf.Session() as sess:
    image_raw_data=cv2.imread(jpgPath)
    #plt.imshow(image_raw_data)
    #plt.show()

    img_data = tf.image.convert_image_dtype(image_raw_data, dtype=tf.float32)

    #img_data = tf.image.resize_images(img_data, [int(1024 * 0.5), int(1024 * 0.5)], method=1)
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
    # 264.0,152.0,213.0,379.0 x,y,width,height [y1,x1 , y2,x2]
    #boxes = tf.constant([[[152.0/1024.0, 264.0/1024.0, (152.0+379.0)/1024.0, (264.0+213.0)/1024.0]]])

    dicom_num = 0
    dicom_x1 = 0
    dicom_y1 = 0
    dicom_width1 = 0
    dicom_height1 = 0
    dicom_x2 = 0
    dicom_y2=0
    dicom_width2=0
    dicom_height2=0
    with open(labelPath,'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if '1' in row.get('Target'):
                if filename in row.get('patientId'):
                    if dicom_num == 0:
                        #x,y,width,height
                        dicom_x1 = string.atof(row.get('x'))
                        dicom_y1 = string.atof(row.get('y'))
                        dicom_width1 = string.atof(row.get('width'))
                        dicom_height1 = string.atof(row.get('height'))
                        dicom_num += 1
                    elif dicom_num == 1:
                        dicom_x2 = string.atof(row.get('x'))
                        dicom_y2 = string.atof(row.get('y'))
                        dicom_width2 = string.atof(row.get('width'))
                        dicom_height2 = string.atof(row.get('height'))
                        dicom_num += 1
                    else:
                        print "dicom_num > 2!"
    boxes = tf.constant([[
        [dicom_y1/1024.0, dicom_x1/1024.0,
            (dicom_y1+dicom_height1)/1024.0, (dicom_x1+dicom_width1)/1024.0],
        [dicom_y2 / 1024.0, dicom_x2 / 1024.0,
            (dicom_y2 + dicom_height2) / 1024.0, (dicom_x2 + dicom_width2) / 1024.0]]])
    result = tf.image.draw_bounding_boxes(batched, boxes)

    result = tf.squeeze(result, 0)
    plt.imshow(result.eval())
    plt.show()
