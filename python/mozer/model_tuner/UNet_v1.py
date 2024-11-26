import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations

def _conv_block(in_data, out_dim):
    """Basic convolution block with batch normalization and LeakyReLU"""
    x = layers.Conv2D(filters=out_dim, kernel_size=3, strides=1, padding='same')(in_data)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)   
    return x

def _conv_trans_block(in_data, out_dim):
    """Transposed convolution block with batch normalization and LeakyReLU"""
    x = layers.Conv2DTranspose(filters=out_dim, kernel_size=3, strides=2, 
                             padding='same', output_padding=1)(in_data)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)   
    return x

def _maxpool(in_data):
    """Max pooling layer"""
    return layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(in_data)

def _double_conv_block(in_data, out_dim):
    """Double convolution block"""
    x = _conv_block(in_data=in_data, out_dim=out_dim)
    x = layers.Conv2D(filters=out_dim, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    return x

def _reduction_block(x, num_filter, mutation_depth):
    """Neural reduction block with optional mutation"""
    x = _double_conv_block(x, num_filter)
    pool = _maxpool(x)
    
    _num_filter = num_filter
    for _ in range(mutation_depth):
        x = _conv_block(in_data=_maxpool(x), out_dim=_num_filter)
        _num_filter //= 2
        
    return x, pool, num_filter * 2

def _expansion_block(up_x, skip_x, num_filter, mutation_depth):
    """Neural expansion block with optional mutation"""
    trans = _conv_trans_block(up_x, num_filter)
    
    _num_filter = num_filter
    for _ in range(mutation_depth):
        _num_filter //= 2
    for _ in range(mutation_depth):
        _num_filter *= 2
        skip_x = _conv_trans_block(skip_x, _num_filter)
        
    x = layers.Concatenate(axis=3)([trans, skip_x])
    x = _double_conv_block(x, num_filter)
    return x, num_filter // 2

def UNet(in_dim, out_dim, num_filter, input_shape=(256,256), mutation=[0, 0, 0, 0]):
    """
    UNet architecture with mutation options for each level
    
    Args:
        in_dim: Number of input channels
        out_dim: Number of output channels
        num_filter: Initial number of filters
        input_shape: Input image dimensions (height, width)
        mutation: List of mutation depths for each level [0-3]
    """
    if len(mutation) != 4:
        raise ValueError("Mutation list must have exactly 4 elements")

    inputs = layers.Input(shape=(*input_shape, in_dim))
    
    # Encoder path
    down1, pool1, num_filter = _reduction_block(inputs, num_filter, mutation[0])
    down2, pool2, num_filter = _reduction_block(pool1, num_filter, mutation[1])
    down3, pool3, num_filter = _reduction_block(pool2, num_filter, mutation[2])
    down4, pool4, num_filter = _reduction_block(pool3, num_filter, mutation[3])
    
    # Bridge
    bridge = _double_conv_block(pool4, num_filter)
    
    # Decoder path
    num_filter = num_filter // 2
    up4, num_filter = _expansion_block(bridge, down4, num_filter, mutation[3])
    up3, num_filter = _expansion_block(up4, down3, num_filter, mutation[2])
    up2, num_filter = _expansion_block(up3, down2, num_filter, mutation[1])
    up1, num_filter = _expansion_block(up2, down1, num_filter, mutation[0])
    
    # Output layer
    outputs = layers.Conv2D(filters=out_dim, kernel_size=3, strides=1,
                          padding='same', activation=activations.sigmoid)(up1)
    
    return keras.models.Model(inputs=inputs, outputs=outputs)