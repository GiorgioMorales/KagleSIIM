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


model = load_model('Redes/Test2.h5', custom_objects={'relu6': relu6, 'focal_loss': focal_loss})

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
Carga imágenes 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
ids = []

images = np.zeros((len(addri), dim, dim, 1), dtype=np.uint8)

for cnt, dir in enumerate(addri):

    # Lee imagen
    ds = pydicom.read_file(dir)
    img = ds.pixel_array

    if dim != 1024:
        img = cv2.resize(img, (dim, dim))

    images[cnt,] = np.reshape(img, (dim, dim, 1))

    # Extrar nombre
    name = os.path.basename(dir)[:-4]
    ids.append(name)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Predecir
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

pred = model.predict(images)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Predice resultados
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('Empieza predicción')

# with open('persons.csv', 'w') as csvfile:
#
#     filewriter = csv.writer(csvfile, delimiter=',')
#     filewriter.writerow(['ImageID', 'EncodedPixels'])
rles = []

for cnt, dir in enumerate(addri):

    print(cnt)

    # Delets small objects
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats((pred[cnt]*255 > 160).astype(np.uint8) * 255,
                                                                               connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 70
    y2 = np.zeros(pred[cnt].shape, dtype=np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            y2[output == i + 1] = 255

    nb_components, _, _, _ = cv2.connectedComponentsWithStats(y2, connectivity=8)

    # Codifica máscara
    mask = (cv2.resize(y2, (1024, 1024)) > 128).astype(np.uint8) * 255
    rles.append(mask2rle(mask, 1024, 1024))

    # filewriter.writerow([name, rle])
import pandas as pd
sub_df = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rles})
sub_df.loc[sub_df.EncodedPixels=='', 'EncodedPixels'] = '-1'
sub_df.to_csv('submission.csv', index=False)
sub_df.head()
