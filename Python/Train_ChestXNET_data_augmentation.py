from random import shuffle
from keras.applications.densenet import DenseNet121
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import load_model, Model
from models_Giorgio import compiled_model, focal_loss
import numpy as np
import cv2
import glob
import os
import pydicom
import keras
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
#from google.colab.patches import cv2_imshow
import pickle



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

  def __data_generation(self, list_IDs_temp):
    'Genera data'  # X : (n_samples, *dim, n_channels)
    # Inicializa input y output
    X = np.empty((self.batch_size, self.dim, self.dim, self.n_channels), dtype=np.float)
    y = np.empty((self.batch_size, 1))

    # Generate data
    for i, ID in enumerate(list_IDs_temp):  # [(0,'C://dataset//im423.jpg'),(1,'C://dataset//im672.jpg'),...]

        addr = self.path + ID[len(self.path):]
        addrm = self.maskpath + ID[len(self.path):]  # Lee dirección de la máscara / label
        img = cv2.imread(addr, 0)
        mask = cv2.imread(addrm, 0)

        # Resize si es necesario
        if self.dim != 1024:
            img = cv2.resize(img, (self.dim, self.dim))
            mask = cv2.resize(mask, (self.dim, self.dim))

        # Lo pone en la forma adecuada
        img = np.reshape(img, (self.dim, self.dim, 1))
        mask = np.reshape((mask / 255), (self.dim, self.dim, 1))
        
        if mask.max() == 0:
            valy = 0
        else:
            valy = 1
        # Guarda muestra
        img3 = cv2.merge([img, img, img])
        X[i,] = np.reshape(img3, (self.dim, self.dim, 3)) / 255.
        # Guarda máscara / labeladdrm
        y[i,] = valy

    return X, y

def flip(x, y):
  x = np.flip(x, 1)
  y = np.flip(y, 1)
  return x, y

def zoom(x, y, zoom, dim = 1024):
  offset = int(dim * zoom / 100.) * 2
  x2 = np.reshape(cv2.resize(x[int(offset / 2):dim - int(offset / 2), int(offset / 2):dim - int(offset / 2), :],(dim, dim)),
                  (dim, dim, 1))
  y2 = np.reshape(cv2.resize(y[int(offset / 2):dim - int(offset / 2), int(offset / 2):dim - int(offset / 2), :],(dim, dim)), 
                  (dim, dim, 1))
  return x2, y2

def DataAugmentation(basepath = "./KagleSIIM", dim=1024):
  print("Inicia el aumento de datos")
  try:
    os.mkdir("{0}/TrainA".format(basepath))
    os.mkdir("{0}/MasksA".format(basepath))
  except:
    print("Ya existen los directorios")
    return "Listo"

  path_data_pattern = "{0}/Train/*.dcm"
  addri = sorted(glob.glob(path_data_pattern.format(basepath)))

  path_data_output = "{0}/TrainA".format(basepath)
  path_mask_output = "{0}/MasksA".format(basepath)

  path = "{0}/Train".format(basepath)
  maskpath = "{0}/Masks".format(basepath)

  for i, ID in enumerate(addri):
    if (i % 1000) == 0:
      print("Elemento {0} de {1}".format(i, len(addri)))
    data_id = ID[len(path):-4]
    addr = "{0}{1}.dcm".format(path, data_id)
    addrm = "{0}{1}.tif".format(maskpath, data_id)
    ds = pydicom.read_file(addr)
    img = ds.pixel_array
    img = img.astype(np.uint8)
    mask = np.flip(np.rot90(cv2.imread(addrm, 0), 3), 1)
    if dim != 1024:
      img = cv2.resize(img, (dim, dim))
      mask = cv2.resize(mask, (dim, dim))
    img = np.reshape(img, (dim, dim, 1))
    mask = np.reshape((mask), (dim, dim, 1))
    if mask.sum() > 0:
      cv2.imwrite("{0}{1}_original.jpg".format(path_data_output,data_id), img)
      cv2.imwrite("{0}{1}_original.jpg".format(path_mask_output,data_id), mask)
      #img_2, mask_2 = zoom(img, mask, 2)
      #cv2.imwrite("{0}{1}_z2.jpg".format(path_data_output,data_id), img_2)
      #cv2.imwrite("{0}{1}_z2.jpg".format(path_mask_output,data_id), mask_2)
      #img_5, mask_5 = zoom(img, mask, 5)
      #cv2.imwrite("{0}{1}_z5.jpg".format(path_data_output,data_id), img_5)
      #cv2.imwrite("{0}{1}_z5.jpg".format(path_mask_output,data_id), mask_5)
      img_f, mask_f = flip(img, mask)
      cv2.imwrite("{0}{1}_f.jpg".format(path_data_output,data_id), img_f)
      cv2.imwrite("{0}{1}_f.jpg".format(path_mask_output,data_id), mask_f)
      #img_2, mask_2 = zoom(img_f, mask_f, 2)
      #cv2.imwrite("{0}{1}_fz2.jpg".format(path_data_output,data_id), img_2)
      #cv2.imwrite("{0}{1}_fz2.jpg".format(path_mask_output,data_id), mask_2)
      #img_5, mask_5 = zoom(img_f, mask_f, 5)
      #cv2.imwrite("{0}{1}_fz5.jpg".format(path_data_output,data_id), img_5)
      #cv2.imwrite("{0}{1}_fz5.jpg".format(path_mask_output,data_id), mask_5)
      

def Train_model():
  # Obtiene dirección del proyecto y de la carpeta que contiene al proyecto
  basepath = os.getcwd()[:-7]

  # Obtiene listas de imágenes de entrenamiento, valdiación y test
  shuffle_data = True
  orig_path = basepath + '/TrainA/*.jpg'

  # Obtiene una lista de las direcciones de las imágenes y sus máscaras
  addri = sorted(glob.glob(orig_path))

  # Reordena aleatoriamente las direcciones por pares
  if shuffle_data:
      shuffle(addri)

  # Divide 90% train, 10% validation
  train_origin = addri[0:int(0.9 * len(addri))]
  val_origin = addri[int(0.9 * len(addri)):]

  # Parametros para la generación de data
  path = basepath + '/TrainA'
  maskpath = basepath + '/MasksA'
  n_channels = 3
  dim = 256
  params = {'dim': dim,
            'batch_size': 12,
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

  # Define función para copiar layers
  def copyModel2Model(model_source, model_target, certain_layer=""):
      for l_tg, l_sr in zip(model_target.layers, model_source.layers):
          wk0 = l_sr.get_weights()
          l_tg.set_weights(wk0)
          l_tg.trainable = False
          print(l_tg.name)
          if l_tg.name == certain_layer:
              break
      print("se copiaron los pesos")


  # Construye una DenseNet121 con una salida sigmoid
  model = DenseNet121(include_top=False, weights=None, input_tensor=None,
                      input_shape=(256, 256, 3), pooling=None, classes=1)
  y = model.get_layer('relu').output
  y = Dropout(0.1)(y)
  y = GlobalAveragePooling2D()(y)
  y = Dense(1, activation='sigmoid', name='Prediction')(y)
  model2 = Model(inputs=model.input, outputs=y)

  # Copia los pesos de la red pre-entrenada
  model_base = load_model('Redes/CheXNet_network.h5', custom_objects={'focal_loss': focal_loss})
  copyModel2Model(model_base, model, "conv5_block16_concat") #pool4_pool

  # Carga la red
  # model2 = load_model('Redes/')

  # Compila la red
  optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
  model2.compile(optimizer=optimizer, loss=focal_loss, metrics=['acc'])
  model2.summary()

  # checkpoint
  filepath = "weights-trainclasschest-{epoch:02d}-{val_acc:.4f}.h5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  callbacks_list = [checkpoint]

  # Train model
  print("Empieza entrenamiento...")
  history = model2.fit_generator(generator=training_generator,
                                validation_data=validation_generator,
                                use_multiprocessing=False,
                                shuffle=True,
                                epochs=10,
                                callbacks=callbacks_list)


if __name__ == '__main__':
  DataAugmentation(basepath = os.getcwd()[:-7])
  Train_model()
