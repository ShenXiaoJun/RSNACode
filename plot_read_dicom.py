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

print(__doc__)

#filename = get_testdata_files('./0a0f91dc-6015-4342-b809-d19610854a21.dcm')[0]
#filename = '/home/shenxj/RSNA/data/RSNA/stage_1_train_images/0a0f91dc-6015-4342-b809-d19610854a21.dcm'
filename = '/home/shenxj/RSNA/data/RSNA/stage_1_train_images/00436515-870c-4b36-a041-de91049b9ab4.dcm'
dataset = pydicom.dcmread(filename)

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

# plot the image using matplotlib
plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
plt.show()
