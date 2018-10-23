"""
=======================================
Read DICOM and ploting using matplotlib
=======================================

This example illustrates how to open a DICOM file, print some dataset
information, and show it using matplotlib.

"""

# authors : Guillaume Lemaitre <g.lemaitre58@gmail.com>
# license : MIT

import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
import tensorflow as tf
import cv2
import numpy as np
import csv

print(__doc__)


#filename = '/home/shenxj/RSNA/data/RSNA/stage_1_train_images/0a0f91dc-6015-4342-b809-d19610854a21.dcm'
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
    boxes = tf.constant([[
        [152.0/1024.0, 264.0/1024.0, (152.0+379.0)/1024.0, (264.0+213.0)/1024.0],
        [152.0/1024.0, 562.0/1024.0, (152.0+453.0)/1024.0, (562.0+256.0)/1024.0]]])




    result = tf.image.draw_bounding_boxes(batched, boxes)

    result = tf.squeeze(result, 0)
    plt.imshow(result.eval())
    plt.show()
