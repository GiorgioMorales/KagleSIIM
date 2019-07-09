# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 16:13:09 2018

@author: Giorgio Morales

Entrenamiento de segmentación de aguajes con Deeplab3+G atrous Depthwise separable convolution
"""

from random import shuffle
import glob
import os
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from models_Giorgio import compiled_model

from dataGenerator import DataGenerator

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

import pickle

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Listas
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Obtiene dirección del proyecto y de la carpeta que contiene al proyecto
basepath = os.getcwd()[:-7]

# Obtiene listas de imágenes de entrenamiento, valdiación y test
shuffle_data = True
orig_path = basepath + '//Train//*.dcm'

# Obtiene una lista de las direcciones de las imágenes y sus máscaras
addri = sorted(glob.glob(orig_path))

# Reordena aleatoriamente las direcciones por pares
if shuffle_data:
    shuffle(addri)

# Divide 90% train, 10% validation
train_origin = addri[0:int(0.9 * len(addri))]
val_origin = addri[int(0.9 * len(addri)):]

# Parametros para la generación de data
path = basepath + '//Train'
maskpath = basepath + '//Masks'
n_channels = 1
dim = 256
params = {'dim': dim,
          'batch_size': 10,
          'n_channels': n_channels,
          'path': path,
          'maskpath': maskpath,
          'shuffle': True}

# Crea diccionarios
data_dict = {}
data_dict["train"] = train_origin
data_dict["validation"] = val_origin

# Generadores
training_generator = DataGenerator(data_dict['train'], **params)
validation_generator = DataGenerator(data_dict['validation'], **params)

# Guarda generadores
# with open('Generators/Train', 'wb') as f:
#     pickle.dump(training_generator, f)
#
# with open('Generators/Validation', 'wb') as f:
#     pickle.dump(validation_generator, f)


# Si los generadores ya han sido creados con anterioridad, sólo se cargan
# with open('Generators/Train', 'rb') as f:
#     training_generator = pickle.load(f)
# with open('Generators/Validation', 'rb') as f:
#     validation_generator = pickle.load(f)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Entrenamiento
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Carga modelo
model = load_model('Redes/Test3.h5')

# Bloquea el entrenamiento desde la primera capa hasta la capa "block5_pool"
for layer in model.layers:
    layer.trainable = False
    if layer.name == 'concatenate_2':
        print('Stop freezing layers')
        break


# checkpoint
filepath = "weights-train_negatives-{epoch:02d}-{val_acc:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]

# Train model
print("Empieza entrenamiento...")
history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              use_multiprocessing=False,
                              shuffle=True,
                              epochs=100,
                              callbacks=callbacks_list)
