from keras.layers import Input, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation
from keras.layers.convolutional import UpSampling2D, Conv2D, DepthwiseConv2D, SeparableConv2D
from keras.layers import Add
from keras.models import Model
from models_Giorgio import compiled_model, focal_loss, dice_coef_metric
from keras.optimizers import Adam
import pydicom
import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import time

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(0)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Carga modelo
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

dim = 256
model = compiled_model('build_clasificator', dim=dim, n_channels = 1, lr = 0.0003, loss = 'focal_loss')
model.load_weights('weights-trainclass-94-0.8172.h5')
optimizer = Adam(lr=0.03, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=optimizer, loss=focal_loss, metrics=['acc'])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Carga imágenes al azar
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
basepath = os.getcwd()[:-7]
orig_path = basepath + '//Train//*.dcm'

# Obtiene una lista de las direcciones de las imágenes y sus máscaras
addri = sorted(glob.glob(orig_path))

# Reordena aleatoriamente las direcciones por pares
shuffle(addri)
#lon = 1000
dirimages = addri
maskpath = basepath + '//Masks//'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Carga imágenes 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
dim = 256
images = np.zeros((len(addri), dim, dim, 1), dtype=np.uint8)
masks = np.zeros((len(addri), 1), dtype=np.float)

for cnt, dir in enumerate(dirimages):

    # Lee imagen
    ds = pydicom.read_file(dir)
    img = ds.pixel_array
    # Dirección de máscara
    dirm = maskpath + os.path.basename(dir)[:-4] + '.tif'
    # Lee máscara
    mask = np.flip(np.rot90(cv2.imread(dirm, 0), 3), 1)
    if mask.max() == 0:
        valy = 0
    else:
        valy = 1

    if dim != 1024:
        img = cv2.resize(img, (dim, dim))

    images[cnt,] = np.reshape(img, (dim, dim, 1))
    masks[cnt,] = valy

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Predecir
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

pred = model.predict(images)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Selecciona threshold
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


#dice for threshold selection
def dice_overall(preds, targs):
    n = preds.shape[0]
    # preds = preds.view(n, -1)
    # targs = targs.view(n, -1)
    # preds = preds.astype('float')
    # targs = targs.astype('float')
    intersect = (preds * targs).sum(-1)
    union = (preds+targs).sum(-1)
    u0 = union==0
    intersect[u0] = 1
    union[u0] = 2
    return 2. * intersect / union


thrs = np.arange(110, 180, 1)
accs = []
for th in thrs:
    preds_m = (pred*255 > 128).astype(np.float)
    acc = np.sum(preds_m*masks) / np.sum(masks)
    accs.append(acc)
accs = np.array(accs)

best_dice = accs.max()
best_thr = thrs[accs.argmax()]

#
# import matplotlib.pyplot as plt
# plt.figure()
# plt.imshow(((pred*255 > best_thr)).astype(np.float)[197,:,:,0])
# plt.show()
#
# plt.figure()
# plt.imshow(masks[197,:,:,0])
# plt.show()