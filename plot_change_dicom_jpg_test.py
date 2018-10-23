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
import os
import datetime


print(__doc__)

dcmPath='/home/shenxj/RSNA/data/RSNA/stage_1_test_images/'
jpgPath='/home/shenxj/RSNA/data/stage_1_test_jpg/'


def convert_file(dcm_file_path, jpg_file_path):
    dicom_img = pydicom.read_file(dcm_file_path)
    img = dicom_img.pixel_array
    scaled_img = cv2.convertScaleAbs(img-np.min(img), alpha=(255.0 / min(np.max(img)-np.min(img), 10000)))
    cv2.imwrite(jpg_file_path, scaled_img)

#convert_file(dcmPath, jpgPath)

starttime = datetime.datetime.now()
list = os.listdir(dcmPath)
for i in range(0,len(list)):
    dcmFilePath = os.path.join(dcmPath,list[i])
    print dcmFilePath
    print i
    ID = dcmFilePath.split('/')[-1].split('.')[0]
    jpgFilePath = jpgPath+ID+'.jpg'
    convert_file(dcmFilePath, jpgFilePath)
endtime = datetime.datetime.now()
print (endtime - starttime).seconds