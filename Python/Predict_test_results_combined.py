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

dim = 256
# model = compiled_model('build_clasificator', dim=dim, n_channels = 1, lr = 0.0003, loss = 'focal_loss')
model = load_model('Redes/CheXNet_network_pretrained.h5', custom_objects={'focal_loss':focal_loss})
model.load_weights('Redes/weights-trainclasschest-10-0.8788.h5')
optimizer = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=optimizer, loss=focal_loss, metrics=['acc'])

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

pred1 = model.predict(images)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Carga modelo
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def relu6(x):
    return K.relu(x, max_value=6)


model2 = load_model('Redes/Test3.h5', custom_objects={'relu6': relu6, 'focal_loss': focal_loss})

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

    if pred1[cnt] <= 0.4313:
        mask = np.zeros((1024, 1024))
    else:
        pred = np.reshape(model2.predict(np.reshape(images[cnt], (1, dim, dim, 1))), (256, 256))
        # Delets small objects
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats((pred*255 > 140).astype(np.uint8) * 255,
                                                                                   connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        min_size = 70
        y2 = np.zeros(pred.shape, dtype=np.uint8)
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                y2[output == i + 1] = 255

        nb_components, _, _, _ = cv2.connectedComponentsWithStats(y2, connectivity=8)

        # Codifica máscara
        mask = (cv2.resize(y2, (1024, 1024)) > 145).astype(np.uint8) * 255

    rles.append(mask2rle(mask, 1024, 1024))

    # filewriter.writerow([name, rle])
import pandas as pd
sub_df = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rles})
sub_df.loc[sub_df.EncodedPixels=='', 'EncodedPixels'] = '-1'
sub_df.to_csv('submission2.csv', index=False)
sub_df.head()
