import os
import pydicom
import glob
import cv2
import numpy as np
from shutil import copyfile, move
from mask_functions import *
import matplotlib.pyplot as plt

# for path, dirs, files in os.walk(inputdir):
#         for d in dirs:
#             for f in glob.iglob(os.path.join(path, d, '*.dcm')):
#                 cnt = cnt + 1
#
#                 # copyfile(f, 'E://Giorgio//Kagle//Test//' + os.path.basename(f))
#
#                 # Read iamge
#                 ds = pydicom.read_file(f)
#                 img = ds.pixel_array
#                 # cv2.imwrite('E://Giorgio//Kagle//Train//image' + str(cnt) + ".tif", img)


inputdir = 'E://Giorgio//Kagle//Train/'
maskdir = 'E://Giorgio//Kagle//Masks/'

# Verifica que todas las imágenes tengan máscara, sino las mueve
# orig_path = 'E://Giorgio//Kagle//Train/*.dcm'
# for dir in glob.glob(orig_path):
#     mdir = os.path.basename(dir)[:-4] + ".tif"
#
#     if not os.path.exists(os.path.join(maskdir, mdir)):
#         move(dir, 'E://Giorgio//Kagle//Unused//' + os.path.basename(dir))

d = '1.2.276.0.7230010.3.1.4.8323329.352.1517875162.525205'
ds = pydicom.read_file(os.path.join(inputdir, d) + '.dcm')
img = ds.pixel_array

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
img2 = clahe.apply(img)

plt.figure()
plt.imshow(img, cmap='gray')
plt.show()

plt.figure()
plt.imshow(img2, cmap='gray')
plt.show()

mask = np.flip(np.rot90(cv2.imread(os.path.join(maskdir, d) + '.tif', 0), 3), 1)

plt.figure()
plt.imshow(mask, cmap='gray')
plt.show()

inputdir = 'C://Users//User//KagleSIIM//Train/'
maskdir = 'C://Users//User//KagleSIIM//MasksO/'
# Crea carpeta sólo para imágenes con pneumothorax
orig_path = 'C://Users//User//KagleSIIM//Train/*.dcm'
for dir in glob.glob(orig_path):
    mdir = os.path.basename(dir)[:-4] + ".tif"

    if os.path.exists(os.path.join(maskdir, mdir)):
        copyfile(dir, 'C://Users//User//KagleSIIM//TrainO//' + os.path.basename(dir))


maskdir = 'C://Users//User//KagleSIIM2//MasksO/*.tif'
# Crea carpeta sólo para imágenes con pneumothorax
orig_path = 'C://Users//User//KagleSIIM2//TrainA/'
mask_path = 'C://Users//User//KagleSIIM2//MasksA/'
for dir in glob.glob(maskdir):
    mdir = os.path.join(mask_path, os.path.basename(dir)[:-4] + "_f.jpg")
    mdir2 = os.path.join(mask_path, os.path.basename(dir)[:-4] + "_original.jpg")
    mdir3 = os.path.join(mask_path, os.path.basename(dir)[:-4] + "_rot5.jpg")
    mdir4 = os.path.join(mask_path, os.path.basename(dir)[:-4] + "_rot355.jpg")

    tdir = os.path.join(orig_path, os.path.basename(dir)[:-4] + "_f.jpg")
    tdir2 = os.path.join(orig_path, os.path.basename(dir)[:-4] + "_original.jpg")
    tdir3 = os.path.join(orig_path, os.path.basename(dir)[:-4] + "_rot5.jpg")
    tdir4 = os.path.join(orig_path, os.path.basename(dir)[:-4] + "_rot355.jpg")

    if os.path.exists(mdir):
        copyfile(mdir, 'C://Users//User//KagleSIIM2//MasksAO//' + os.path.basename(mdir))
        copyfile(mdir2, 'C://Users//User//KagleSIIM2//MasksAO//' + os.path.basename(mdir2))
        copyfile(mdir3, 'C://Users//User//KagleSIIM2//MasksAO//' + os.path.basename(mdir3))
        copyfile(mdir4, 'C://Users//User//KagleSIIM2//MasksAO//' + os.path.basename(mdir4))

        copyfile(tdir, 'C://Users//User//KagleSIIM2//TrainAO//' + os.path.basename(tdir))
        copyfile(tdir2, 'C://Users//User//KagleSIIM2//TrainAO//' + os.path.basename(tdir2))
        copyfile(tdir3, 'C://Users//User//KagleSIIM2//TrainAO//' + os.path.basename(tdir3))
        copyfile(tdir4, 'C://Users//User//KagleSIIM2//TrainAO//' + os.path.basename(tdir4))

