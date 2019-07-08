from keras.models import Model, load_model
from models_Giorgio import compiled_model, focal_loss
from keras.optimizers import Adam
import pydicom
import cv2
import os
import glob
import numpy as np
import time

import csv
from mask_functions import *

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(0)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Carga modelo
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def relu6(x):
    return K.relu(x, max_value=6)


model = load_model('Redes/Test1.h5', custom_objects={'relu6': relu6, 'focal_loss': focal_loss})

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Carga imágenes test
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

basepath = os.getcwd()[:-7]
orig_path = basepath + '//Test//*.dcm'

# Obtiene una lista de las direcciones de las imágenes y sus máscaras
addri = sorted(glob.glob(orig_path))

# Define directorio de máscaras de test
maskpath = basepath + '//MasksTest//'

dim = 256
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Predice resultados
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('Empieza predicción')

with open('persons.csv', 'w') as csvfile:

    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(['ImageID', 'EncodedPixels'])

    for cnt, dir in enumerate(addri):

        # Lee imagen
        ds = pydicom.read_file(dir)
        img = ds.pixel_array
        # Extrar nombre
        name = os.path.basename(dir)[:-4]

        if dim != 1024:
            img = cv2.resize(img, (dim, dim))

        # Predice resultado
        st = time.time()
        y = np.reshape(model.predict(np.reshape(img, (1, dim, dim, 1))), (dim, dim)) * 255
        end = time.time()

        print(end - st)

        # Delets small objects
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats((y > 128).astype(np.uint8) * 255,
                                                                                   connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        min_size = 70
        y2 = np.zeros(y.shape)
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                y2[output == i + 1] = 1

        # Codifica máscara
        mask = (cv2.resize(y2, (1024, 1024)) > 128).astype(np.uint8) * 255
        rle = mask2rle(mask, 1024, 1024)

        filewriter.writerow([name, rle])
