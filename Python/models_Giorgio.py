
from keras.layers import Input, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation
from keras.layers.convolutional import UpSampling2D, Conv2D, DepthwiseConv2D, SeparableConv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Add
import tensorflow as tf

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Custom functions
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def focal_loss(y_true, y_pred, gamma=2., alpha=.75):

    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    epsilon = K.epsilon()

    # clip to prevent NaN's and Inf's
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Modelo 1
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def build_generator1(img_shape=(1024, 1024, 1)):

    def relu6(x):
        return K.relu(x, max_value=6)

    def _inverted_res_block(inputs, expansion, stride, pointwise_filters, block_id, skip_connection, rate=1):

        in_channels = inputs._keras_shape[-1]

        x = inputs
        prefix = 'expanded_conv_{}_'.format(block_id)

        if block_id:
            # Expand

            x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                       use_bias=False, activation=None,
                       name=prefix + 'expand')(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                                   name=prefix + 'expand_BN')(x)
            x = Activation(relu6, name=prefix + 'expand_relu')(x)
        else:
            prefix = 'expanded_conv_'
        # Depthwise
        x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                            use_bias=False, padding='same', dilation_rate=(rate, rate),
                            name=prefix + 'depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'depthwise_BN')(x)

        x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

        # Project
        x = Conv2D(pointwise_filters,
                   kernel_size=1, padding='same', use_bias=False, activation=None,
                   name=prefix + 'project')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'project_BN')(x)

        if skip_connection:
            return Add(name=prefix + 'add')([inputs, x])

        return x

    # Image input
    d0 = Input(shape=img_shape)

    """""""""""""""""""""""""""
    """"""""""""""""""""""""""
    Feature extraction
    """"""""""""""""""""""""""
    """""""""""""""""""""""""""

    first_block_filters = 32
    x = Conv2D(first_block_filters,
               kernel_size=3,
               strides=(2, 2), padding='same',
               use_bias=False, name='Conv')(d0)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    x = Activation(relu6, name='Conv_Relu6')(x)

    x = _inverted_res_block(x, pointwise_filters=16, stride=2,
                            expansion=1, block_id=0, skip_connection=False)

    skip1 = x

    x = _inverted_res_block(x, pointwise_filters=24, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    x = _inverted_res_block(x, pointwise_filters=24, stride=1,
                            expansion=6, block_id=2, skip_connection=True)

    x = _inverted_res_block(x, pointwise_filters=32, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    x = _inverted_res_block(x, pointwise_filters=32, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    x = _inverted_res_block(x, pointwise_filters=32, stride=1,
                            expansion=6, block_id=5, skip_connection=True)

    """""""""""""""""""""""""""
    """"""""""""""""""""""""""
    ASPP
    """"""""""""""""""""""""""
    """""""""""""""""""""""""""
    atrous_rates = (6, 12, 18)

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

    # concatenate ASPP branches and project
    x = Concatenate()([b0, b1, b2, b3])
    b0 = None
    b1 = None
    b2 = None
    b3 = None

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

    dec_skip1 = None

    x = SeparableConv2D(256, kernel_size=3, strides=(1, 1), padding='same',
                        dilation_rate=1, name='decoder_conv0')(x)
    x = SeparableConv2D(256, kernel_size=3, strides=(1, 1), padding='same',
                        dilation_rate=1, name='decoder_conv1')(x)

    x = Conv2D(1, (1, 1), padding='same', name="last_layer", activation='sigmoid')(x)
    x = UpSampling2D(size=4)(x)

    return Model(d0, x)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Compile model
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def compiled_model (modelname = 'build_generator1', dim =1024, n_channels = 1, lr = 0.03, loss = 'focal_loss'):

    loss = globals()[loss]
    print("Cargando modelo...")

    model = globals()[modelname]((dim, dim, n_channels))
    model.summary()

    # Compila modelol
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc', dice_coef_loss])

    return model
