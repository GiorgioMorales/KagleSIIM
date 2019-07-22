# -*- coding: utf-8 -*-
"""
GENERADOR DE DATA EN KERAS
"""
import numpy as np
import keras
import cv2
import random
import pydicom
import os


class DataGenerator(keras.utils.Sequence):
    'Inicializa variables'

    def __init__(self, list_IDs, batch_size=1, dim=512, n_channels=3, path='', maskpath='',
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.path = path
        self.maskpath = maskpath
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Calcula el número de batches por época'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Genera un batch'
        # Genera los índices del batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Crea una lista de IDs correspondientes a indexes
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Genera la data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Actualiza indexes'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def randomflip(self, x, y):
        p = random.randint(0, 1)
        if p:
            x = np.flip(x, 1)
            y = np.flip(y, 1)
        return x, y

    def randomclahe(self, x):
        p = random.randint(0, 1)
        if p:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
            x = (clahe.apply(x))
        return x.astype(np.uint8)

    def randomzoom(self, x, y, zoom):
        p = random.randint(0, zoom)
        offset = int(self.dim * p / 100.) * 2
        x2 = cv2.resize(x[int(offset / 2):self.dim - int(offset / 2), int(offset / 2):self.dim - int(offset / 2), :],
                        (self.dim, self.dim))
        y2 = np.reshape(cv2.resize(y[int(offset / 2):self.dim - int(offset / 2), int(offset / 2):self.dim - int(offset / 2), :],
                        (self.dim, self.dim)), (self.dim, self.dim, 1))
        return x2, y2

    def __data_generation(self, list_IDs_temp):
        'Genera data'  # X : (n_samples, *dim, n_channels)
        # Inicializa input y output
        X = np.empty((self.batch_size, self.dim, self.dim, self.n_channels), dtype=np.float)
        y = np.empty((self.batch_size, self.dim, self.dim, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):  # [(0,'C://dataset//im423.jpg'),(1,'C://dataset//im672.jpg'),...]

            addr = self.path + ID[len(self.path):]

            if ID[-4:] == '.dcm':
                # Lee imagen DICOM
                ds = pydicom.read_file(addr)
                img = ds.pixel_array
                # Lee máscara
                addrm = self.maskpath + ID[len(self.path):-4] + '.tif'  # Lee dirección de la máscara / label
                mask = np.flip(np.rot90(cv2.imread(addrm, 0), 3), 1)
            else:
                # Lee imagen JPG
                img = cv2.imread(addr, 0)
                # Lee máscara
                addrm = self.maskpath + ID[len(self.path):-4] + '.jpg'  # Lee dirección de la máscara / label
                mask = cv2.imread(addrm, 0)

            # Añade CLAHE (Contrast Limited Adaptive Histogram Equalization)
            img2 = self.randomclahe(img)

            # Resize si es necesario
            if self.dim != 1024:
                img2 = cv2.resize(img2, (self.dim, self.dim))
                mask = cv2.resize(mask, (self.dim, self.dim))

            # Lo pone en la forma adecuada
            img2 = np.reshape(img2, (self.dim, self.dim, 1))
            mask2 = np.reshape((mask / 255), (self.dim, self.dim, 1))
            mask2[mask2 > 0] = 1

            # # Añade variación aleatoria de color a cada canal
            # img2, mask2 = self.randomflip(img2, mask2)
            # img2, mask2 = self.randomzoom(img2, mask2, 5)

            # Guarda muestra
            img3 = cv2.merge([img2, img2, img2])
            X[i, ] = np.reshape(img3, (self.dim, self.dim, 3)) / 255.
            # Guarda máscara / labeladdrm
            y[i, ] = mask2

        return X, y
