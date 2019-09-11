# BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs (https://arxiv.org/abs/1907.05047)
## but it's "not same architecture" as shown in the above.

import tensorflow as tf
import tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Add, ReLU, MaxPooling2D, Reshape, Lambda, Activation, DepthwiseConv2D
import numpy as np

def conv_blocks(x, num_filter, pooling=False, channel_padding=False, pad_value=None):
    x = ReLU()(x)
    shortcut = x
    
    if pooling:
        shortcut = MaxPooling2D(strides=(2, 2), padding='same')(shortcut)
    if channel_padding:
        paddings = tf.constant([[0,0],[0,0],[0,0],[0,pad_value]])
        shortcut = Lambda(lambda x: tf.pad(x, paddings, "CONSTANT"))(shortcut)
    
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(x)
    x = Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
    x = Add()([x, shortcut])
    return x

def face_detection_model(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    x = Conv2D(24, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=True)(x) # (1, 64, 64, 24)
    x = conv_blocks(x, 24, pooling=False, channel_padding=False) # (1, 64, 64, 24)
    x = conv_blocks(x, 28, pooling=False, channel_padding=True, pad_value=4) # (1, 64, 64, 28)
    x = conv_blocks(x, 32, pooling=True, channel_padding=True, pad_value=4) # (1, 32, 32, 32)
    x = conv_blocks(x, 36, pooling=False, channel_padding=True, pad_value=4) # (1, 32, 32, 36)
    x = conv_blocks(x, 42, pooling=False, channel_padding=True, pad_value=6) # (1, 32, 32, 42)
    x = conv_blocks(x, 48, pooling=True, channel_padding=True, pad_value=6) # (1, 16, 16, 48)
    x = conv_blocks(x, 56, pooling=False, channel_padding=True, pad_value=8) # (1, 16, 16, 56)
    x = conv_blocks(x, 64, pooling=False, channel_padding=True, pad_value=6) # (1, 16, 16, 64)
    x = conv_blocks(x, 72, pooling=False, channel_padding=True, pad_value=8) # (1, 16, 16, 72)
    x = conv_blocks(x, 80, pooling=False, channel_padding=True, pad_value=8) # (1, 16, 16, 80)
    x = conv_blocks(x, 88, pooling=False, channel_padding=True, pad_value=8) # (1, 16, 16, 88)

    x = ReLU()(x)
    shortcut_1 = x # (1, 16, 16, 88)
    shortcut = x

    shortcut = MaxPooling2D(strides=(2, 2), padding='same')(shortcut)
    paddings = tf.constant([[0,0],[0,0],[0,0],[0,8]]) # pad_value = 8
    shortcut = Lambda(lambda x: tf.pad(x, paddings, "CONSTANT"))(shortcut)

    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True)(x)
    x = Conv2D(num_filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)(x)
    x = Add()([x, shortcut]) # (1, 8, 8, 96)

    x = conv_blocks(x, 96, pooling=False, channel_padding=False) # (1, 8, 8, 96)
    x = conv_blocks(x, 96, pooling=False, channel_padding=False) # (1, 8, 8, 96)
    x = conv_blocks(x, 96, pooling=False, channel_padding=False) # (1, 8, 8, 96)
    x = conv_blocks(x, 96, pooling=False, channel_padding=False) # (1, 8, 8, 96)

    # Last layer
    x = ReLU()(x)
    ## Classificators
    classificator_8 = Conv2D(2, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)(shortcut_1)
    classificator_16 = Conv2D(6, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)(x)
    ### Classificators: Reshape and Concatenation
    classificator_8 = Reshape(target_size=(-1,1))(classificator_8) # (1, 512, 1)
    classificator_16 = Reshape(target_size=(-1,1))(classificator_16) # (1, 384, 1)
    classificator_concat = Concatenate(axis=-1, name='classificators')([classificator_8, classificator_16])

    ## Regressors
    regressor_8 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)(shortcut_1)
    regressor_16 = Conv2D(96, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)(x)
    ### Regressors: Reshape and Concatenation
    regressor_8 = Reshape(target_size=(-1,1))(regressor_8) # (1, 512, 1)
    regressor_16 = Reshape(target_size=(-1,1))(regressor_16) # (1, 384, 1)
    regressor_concat = Concatenate(axis=-1, name='regressors')([regressor_8, regressor_16])

    model = Model(inputs, [regressor_concat, classificator_concat])
    return model
