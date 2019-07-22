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
n_channels = 3
# model = compiled_model('build_clasificator', dim=dim, n_channels = 1, lr = 0.0003, loss = 'focal_loss')
model = load_model('Redes/CheXNet_network_pretrained.h5', custom_objects={'focal_loss':focal_loss})
model.load_weights('Redes/weights-trainclasschest-219-0.8570.h5')
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

images = np.zeros((len(addri), dim, dim, n_channels), dtype=np.float)

for cnt, dir in enumerate(addri):

    # Lee imagen
    ds = pydicom.read_file(dir)
    img = ds.pixel_array
    if n_channels == 3:
        img = cv2.merge([img, img, img])

    if dim != 1024:
        img = cv2.resize(img, (dim, dim),interpolation=cv2.INTER_AREA)

    images[cnt, ] = np.reshape(img, (dim, dim, n_channels)) / 255.

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

model2 = compiled_model('UEfficientNet', dim=dim, n_channels = 3, lr = 0.0003, loss = 'focal_loss')
model2.load_weights('Redes/weights-train1-01-0.8053.h5')
optimizer = Adam(lr=0.03, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model2.compile(optimizer=optimizer, loss=focal_loss, metrics=['acc'])
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

    if pred1[cnt] <= 0.78:
        mask = np.zeros((1024, 1024))
    else:
        pred = np.reshape(model2.predict(np.reshape(images[cnt] * 255, (1, dim, dim, n_channels))), (dim, dim))
        # Delets small objects
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats((pred*255 > 128).astype(np.uint8) * 255,
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
        mask = (cv2.resize(y2, (1024, 1024)) > 50).astype(np.uint8) * 255

    rles.append(mask2rle(mask, 1024, 1024))

    # filewriter.writerow([name, rle])
import pandas as pd
sub_df = pd.DataFrame({'ImageId': ids, 'EncodedPixels': rles})
sub_df.loc[sub_df.EncodedPixels=='', 'EncodedPixels'] = '-1'
sub_df.to_csv('submission3.csv', index=False)
sub_df.head()
