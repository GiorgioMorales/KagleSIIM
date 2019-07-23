"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import cv2
import keras
import pydicom
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Conv2DTranspose, UpSampling2D, SeparableConv2D
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras import backend as K
from keras.layers import LeakyReLU
from keras.losses import binary_crossentropy
import keras.callbacks as callbacks
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.merge import concatenate
import glob
import random

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Set Folders
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

seed = 10
np.random.seed(seed)
random.seed(seed)

train_im_path, train_mask_path = '../keras_im_train', '../keras_mask_train'
h, w, batch_size = 256, 256, 4

val_im_path, val_mask_path = '../keras_im_val', '../keras_mask_val'

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Generator
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, train_im_path=train_im_path, train_mask_path=train_mask_path,
                 augmentations=None, batch_size=batch_size, img_size=256, n_channels=3, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.train_im_paths = glob.glob(train_im_path + '/*')

        self.train_im_path = train_im_path
        self.train_mask_path = train_mask_path

        self.img_size = img_size

        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augmentations
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.train_im_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:min((index + 1) * self.batch_size, len(self.train_im_paths))]

        # Find list of IDs
        list_IDs_im = [self.train_im_paths[k] for k in indexes]

        # Generate data
        X, y = self.data_generation(list_IDs_im)

        if self.augment is None:
            return X, np.array(y) / 255
        else:
            im, mask = [], []
            for x, y in zip(X, y):
                augmented = self.augment(image=x, mask=y)
                im.append(augmented['image'])
                mask.append(augmented['mask'])
            return np.array(im), np.array(mask) / 255

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.train_im_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, list_IDs_im):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(list_IDs_im), self.img_size, self.img_size, self.n_channels))
        y = np.empty((len(list_IDs_im), self.img_size, self.img_size, 1))

        # Generate data
        for i, im_path in enumerate(list_IDs_im):

            # im = np.array(Image.open(im_path))
            ds = pydicom.read_file(im_path)
            im = ds.pixel_array
            mask_path = im_path.replace(self.train_im_path, self.train_mask_path)[:-4] + ".tif"

            # mask = np.array(Image.open(mask_path))
            mask = np.flip(np.rot90(cv2.imread(mask_path, 0), 3), 1)
            if len(im.shape) == 2:
                im = np.repeat(im[..., None], 3, 2)

            #             # Resize sample
            X[i,] = cv2.resize(im, (self.img_size, self.img_size))

            # Store class
            y[i,] = cv2.resize(mask, (self.img_size, self.img_size))[..., np.newaxis]
            y[y > 0] = 255

        return np.uint8(X), np.uint8(y)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Data Augmentation
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, OneOf,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, GaussNoise, OpticalDistortion, RandomSizedCrop
)

AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=0.5),
    OneOf([
        RandomContrast(),
        RandomGamma(),
        RandomBrightness(),
    ], p=0.3),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=2, shift_limit=0.5),
    ], p=0.3),
    RandomSizedCrop(min_max_height=(128, 256), height=h, width=w, p=0.5),
    ToFloat(max_value=1)
], p=1)

AUGMENTATIONS_TEST = Compose([
    ToFloat(max_value=1)
], p=1)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Metrics and Losses
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def get_iou_vector(A, B):
    # Numpy version
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)

        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue

        # non empty mask case.  Union is never empty
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union

        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45) * 20)) / 10

        metric += iou

    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))


def focal_loss(y_true, y_pred, gamma=2., alpha=.9):

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    epsilon = K.epsilon()

    # clip to prevent NaN's and Inf's
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1-alpha) * K.pow(pt_0, gamma) *
                                                                         K.log(1. - pt_0))


def dice_coef_metric(y_true, y_pred):

    y_pred = K.greater_equal(y_pred, 0.5)
    y_true = K.greater_equal(y_true, 0.5)
    intersection = K.sum(K.cast(tf.math.logical_and(y_true, y_pred), dtype = tf.float32), axis=[1, 2, 3])
    union = K.sum(K.cast(tf.math.logical_or(y_true, y_pred), dtype = tf.float32), axis=[1, 2, 3])
    dicev = (2. * intersection + K.epsilon()) / (union + intersection + K.epsilon())

    return K.mean(dicev)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Callbacks
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def get_callbacks(self, model_prefix='Model'):

        callback_list = [
            callbacks.ModelCheckpoint("../keras.model",monitor='val_dice_coef_metric:.4f',
                                   mode = 'min', save_best_only=True, verbose=1),
            swa,
            callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                Model
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from efficientnet import EfficientNetB4


def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x


def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = Add()([x, blockInput])
    return x


def UEfficientNet(input_shape=(None, None, 3), dropout_rate=0.1):
    backbone = EfficientNetB4(weights='imagenet',
                              include_top=False,
                              input_shape=input_shape)
    input = backbone.input
    start_neurons = 16

    conv4 = backbone.layers[342].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm, start_neurons * 32)
    convm = residual_block(convm, start_neurons * 32)
    convm = LeakyReLU(alpha=0.1)(convm)

    deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout_rate)(uconv4)

    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 16)
    uconv4 = residual_block(uconv4, start_neurons * 16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)

    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    conv3 = backbone.layers[154].output
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(dropout_rate)(uconv3)

    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 8)
    uconv3 = residual_block(uconv3, start_neurons * 8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    conv2 = backbone.layers[92].output
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 4)
    uconv2 = residual_block(uconv2, start_neurons * 4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[30].output
    uconv1 = concatenate([deconv1, conv1])

    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 2)
    uconv1 = residual_block(uconv1, start_neurons * 2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)

    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0, start_neurons * 1)
    uconv0 = residual_block(uconv0, start_neurons * 1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)

    uconv0 = Dropout(dropout_rate / 2)(uconv0)
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv0)

    model = Model(input, output_layer)
    model.name = 'u-xception'

    return model


def build_generator_combined(img_shape=(256, 256, 3)):

    from keras.applications.densenet import DenseNet121

    # Construye una DenseNet121 con una salida sigmoid
    model = DenseNet121(include_top=False, weights=None, input_tensor=None,
                        input_shape=img_shape, pooling=None, classes=1)
    x = model.get_layer('pool3_conv').output

    skip1 = model.get_layer('conv1/relu').output

    """""""""""""""""""""""""""
    """"""""""""""""""""""""""
    ASPP
    """"""""""""""""""""""""""
    """""""""""""""""""""""""""
    atrous_rates = (3, 6, 9)

    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # rate = 6 (12)
    b1 = SeparableConv2D(256, kernel_size=3, strides=(1, 1), padding='same',
                         dilation_rate=atrous_rates[0], name='app1')(x)
    # rate = 12 (24)
    b2 = SeparableConv2D(256, kernel_size=3, strides=(1, 1), padding='same',
                         dilation_rate=atrous_rates[1], name='app2')(x)
    # rate = 18 (36)
    b3 = SeparableConv2D(256, kernel_size=3, strides=(1, 1), padding='same',
                         dilation_rate=atrous_rates[2], name='app3')(x)

    # Concatenate() ASPP branches and project
    x = Concatenate()([b0, b1, b2, b3])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    """""""""""""""""""""""""""
    """"""""""""""""""""""""""
    Decoder
    """"""""""""""""""""""""""
    """""""""""""""""""""""""""

    x = UpSampling2D(size=4)(x)
    dec_skip1 = Conv2D(48, (1, 1), padding='same',
                       use_bias=False, name='feature_projection0')(skip1)
    dec_skip1 = BatchNormalization(
        name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
    dec_skip1 = Activation('relu')(dec_skip1)
    x = Concatenate()([x, dec_skip1])


    x = SeparableConv2D(256, kernel_size=3, strides=(1, 1), padding='same',
                        dilation_rate=1, name='decoder_conv0')(x)
    x = SeparableConv2D(256, kernel_size=3, strides=(1, 1), padding='same',
                        dilation_rate=1, name='decoder_conv1')(x)

    x = Conv2D(1, (1, 1), padding='same', name="last_layer", activation='sigmoid')(x)
    x = UpSampling2D(size=2)(x)

    return Model(model.input, x)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Compile and transfer learning
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
K.clear_session()
img_size = 256


def copyModel2Model(model_source, model_target, certain_layer="", nfreeze = True):
    for l_tg, l_sr in zip(model_target.layers, model_source.layers):
        wk0 = l_sr.get_weights()
        l_tg.set_weights(wk0)
        l_tg.trainable = nfreeze
        print(l_tg.name)
        if l_tg.name == certain_layer:
            break
    print("se copiaron los pesos")

# Carga modelo
model = build_generator_combined()

# Copia los pesos de la red pre-entrenada
model_base = load_model('Redes/CheXNet_network.h5', custom_objects={'focal_loss': focal_loss})
copyModel2Model(model_base, model, "pool3_conv")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
SWA
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class SWA(keras.callbacks.Callback):

    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch

    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))

    def on_epoch_end(self, epoch, logs=None):

        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()

        elif epoch > self.swa_epoch:
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] *
                                       (epoch - self.swa_epoch) + self.model.get_weights()[i]) / (
                                                  (epoch - self.swa_epoch) + 1)

        else:
            pass

    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print('Final stochastic averaged weights saved to file.')


model.compile(loss=bce_dice_loss, optimizer='adam', metrics=[my_iou_metric, dice_coef_metric])

epochs = 70
snapshot = SnapshotCallbackBuilder(nb_epochs=epochs,nb_snapshots=1,init_lr=1e-3)
swa = SWA('../keras_swa.model',67)
valid_im_path,valid_mask_path = '../keras_im_val','../keras_mask_val'
# Generators
training_generator = DataGenerator(augmentations=AUGMENTATIONS_TRAIN,img_size=img_size, batch_size=batch_size)
validation_generator = DataGenerator(train_im_path = valid_im_path ,
                                     train_mask_path=valid_mask_path,augmentations=AUGMENTATIONS_TEST,
                                     img_size=img_size)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Training
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

history = model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            use_multiprocessing=False,
                            epochs=epochs,verbose=2,
                            callbacks=snapshot.get_callbacks())
