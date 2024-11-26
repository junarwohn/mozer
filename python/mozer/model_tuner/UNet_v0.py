import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Normalization, CenterCrop, Rescaling
from tensorflow.keras import layers, activations
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def _conv_block(in_data, filters, kernel_size=3, strides=1):
    """Basic convolution block with batch normalization and leaky ReLU"""
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, 
                     strides=strides, padding='same')(in_data)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)   
    return x

def _conv_trans_block(in_data, filters, kernel_size=3, strides=2):
    """Transposed convolution block with batch normalization and leaky ReLU"""
    x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                             strides=strides, padding='same', output_padding=1)(in_data)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)   
    return x

def _maxpool():
    """Max pooling layer with fixed parameters"""
    return layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')

def _double_conv_block(in_data, filters):
    """Double convolution block used in UNet"""
    x = _conv_block(in_data, filters)
    x = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x

def UNet(in_dim, out_dim, num_filter, input_size=256):
    """
    UNet architecture implementation
    Args:
        in_dim: Number of input channels
        out_dim: Number of output channels 
        num_filter: Base number of filters
        input_size: Input image size (default 256x256)
    """
    # Input layer
    inputs = layers.Input(shape=(input_size, input_size, in_dim))
    
    # Encoder path
    down_1 = _double_conv_block(inputs, num_filter)
    pool_1 = _maxpool()(down_1)
    
    down_2 = _double_conv_block(pool_1, num_filter * 2)
    pool_2 = _maxpool()(down_2)
    
    down_3 = _double_conv_block(pool_2, num_filter * 4)
    pool_3 = _maxpool()(down_3)
    
    down_4 = _double_conv_block(pool_3, num_filter * 8)
    pool_4 = _maxpool()(down_4)

    # Bridge
    bridge = _double_conv_block(pool_4, num_filter * 16)

    # Decoder path
    trans_1 = _conv_trans_block(bridge, num_filter * 8)
    concat_1 = layers.Concatenate(axis=3)([trans_1, down_4])
    up_1 = _double_conv_block(concat_1, num_filter * 8)

    trans_2 = _conv_trans_block(up_1, num_filter * 4)
    concat_2 = layers.Concatenate(axis=3)([trans_2, down_3])
    up_2 = _double_conv_block(concat_2, num_filter * 4)

    trans_3 = _conv_trans_block(up_2, num_filter * 2)
    concat_3 = layers.Concatenate(axis=3)([trans_3, down_2])
    up_3 = _double_conv_block(concat_3, num_filter * 2)

    trans_4 = _conv_trans_block(up_3, num_filter)
    concat_4 = layers.Concatenate(axis=3)([trans_4, down_1])
    up_4 = _double_conv_block(concat_4, num_filter)

    # Output layer
    outputs = layers.Conv2D(filters=out_dim, kernel_size=3, strides=1, 
                          padding='same', activation='sigmoid')(up_4)

    return keras.models.Model(inputs=inputs, outputs=outputs)