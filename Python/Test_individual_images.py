
from models_Giorgio import compiled_model
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
# Carga modelo
model = compiled_model('build_generator1', loss='focal_loss')

model.load_weights('Redes/weights-train1-01-0.9966.h5')

optimizer = Adam(lr=0.03, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])

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

dirimages = addri[0:3]
maskpath = basepath + '//Masks//'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Dibuja resultados
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('Empieza predicción')
fig, axs = plt.subplots(3, 3)
for cnt, dir in enumerate(dirimages):

    # Lee imagen
    ds = pydicom.read_file(dir)
    img = ds.pixel_array
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    img = (clahe.apply(img)).astype(np.uint8)
    # Dirección de máscara
    dirm = maskpath + os.path.basename(dir)[:-4] + '.tif'
    # Lee máscara
    mask = np.flip(np.rot90(cv2.imread(dirm, 0), 3), 1)
    # Predice resultado
    st = time.time()
    y = np.reshape(model.predict(np.reshape(img, (1, 1024, 1024, 1))), (1024, 1024)) * 255
    end = time.time()

    print(end - st)

    # plt
    titles = ['Original', 'Segmentation', 'Ground truth']
    axs[0, cnt].imshow(img)
    axs[0, cnt].set_title(titles[0])
    axs[0, cnt].axis('off')
    axs[1, cnt].imshow(y)
    axs[1, cnt].set_title(titles[1])
    axs[1, cnt].axis('off')
    axs[2, cnt].imshow(mask)
    axs[2, cnt].set_title(titles[2])
    axs[2, cnt].axis('off')
