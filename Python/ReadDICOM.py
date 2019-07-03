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
