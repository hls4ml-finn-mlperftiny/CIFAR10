import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Add
from tensorflow.keras.regularizers import l1_l2
from qkeras.qlayers import QDense, QActivation
from qkeras.qconvolutional import QConv2D
from qkeras.qpooling import QAveragePooling2D
from qkeras.quantizers import quantized_bits, quantized_relu

#define model
def resnet_v1_eembc(input_shape=[32, 32, 3], num_classes=10, l1p=0, l2p=1e-4,
                    num_filters=[16, 16, # block 1
                                 32, 32, # block 2
                                 64, 64 # block 3
                             ],
                    kernel_sizes=[3, 3, 3, # block 1
                                  3, 3, 1, # block 2
                                  3, 3, 1 # block 3
                              ],
                    strides=['111', # block 1
                             '212', # block 2
                             '212', # block 3
                         ]):

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters[0],
                  kernel_size=kernel_sizes[0],
                  strides=int(strides[0][0]),
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # First stack
    # Weight layers
    y = Conv2D(num_filters[1],
                  kernel_size=kernel_sizes[1],
                  strides=int(strides[0][1]),
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters[0],
                  kernel_size=kernel_sizes[2],
                  strides=int(strides[0][2]),
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(y)
    y = BatchNormalization()(y)
  
    # Overall residual, connect weight layer and identity paths
    x = Add()([x, y]) 
    x = Activation('relu')(x)

    if len(num_filters) > 2 and num_filters[2] > 0 and strides[1] != '' and kernel_sizes[3] > 0:
        # Second stack
        # Weight layers
        y = Conv2D(num_filters[2],
                   kernel_size=kernel_sizes[3],
                   strides=int(strides[1][0]),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)  
        y = Conv2D(num_filters[3],
                   kernel_size=kernel_sizes[4],
                   strides=int(strides[1][1]),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(y)
        y = BatchNormalization()(y)
        
        # Adjust for change in dimension due to stride in identity
        x = Conv2D(num_filters[3],
                   kernel_size=kernel_sizes[5],
                   strides=int(strides[1][2]),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)
        
        # Overall residual, connect weight layer and identity paths
        x = Add()([x, y])
        x = Activation('relu')(x)
        
    if len(num_filters) > 4 and num_filters[4] > 0 and strides[2] != '' and kernel_sizes[6] > 0:
        # Third stack
        # Weight layers
        y = Conv2D(num_filters[4],
                   kernel_size=kernel_sizes[6],
                   strides=int(strides[2][0]),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Conv2D(num_filters[5],
                   kernel_size=kernel_sizes[7],
                   strides=int(strides[2][1]),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(y)
        y = BatchNormalization()(y)
        
        # Adjust for change in dimension due to stride in identity
        x = Conv2D(num_filters[5],
                   kernel_size=kernel_sizes[8],
                   strides=int(strides[2][2]),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)
        
        # Overall residual, connect weight layer and identity paths
        x = Add()([x, y])
        x = Activation('relu')(x)
        
    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)
    y = Dense(num_classes,
                    kernel_initializer='he_normal')(y)
    outputs = Activation('softmax', name='softmax')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

#quantized model
def resnet_v1_eembc_quantized(input_shape=[32, 32, 3], num_classes=10, l1p=0, l2p=1e-4,
                    num_filters=[16, 16, # block 1
                                 32, 32, # block 2
                                 64, 64 # block 3
                             ],
                    kernel_sizes=[3, 3, 3, # block 1
                                  3, 3, 1, # block 2
                                  3, 3, 1 # block 3
                              ],
                    strides=['111', # block 1
                             '212', # block 2
                             '212', # block 3
                         ],
                    logit_total_bits=7, logit_int_bits=0, activation_total_bits=7, activation_int_bits=3):

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    inputs = Input(shape=input_shape)
    x = QConv2D(num_filters[0],
                  kernel_size=kernel_sizes[0],
                  strides=int(strides[0][0]),
                  padding='same',
                  kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(inputs)
    x = BatchNormalization()(x)
    x = QActivation(activation=quantized_relu(activation_total_bits, activation_int_bits))(x)

    # First stack
    # Weight layers
    y = QConv2D(num_filters[1],
                  kernel_size=kernel_sizes[1],
                  strides=int(strides[0][1]),
                  padding='same',
                  kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)
    y = BatchNormalization()(y)
    y = QActivation(activation=quantized_relu(activation_total_bits, activation_int_bits))(y)
    y = QConv2D(num_filters[0],
                  kernel_size=kernel_sizes[2],
                  strides=int(strides[0][2]),
                  padding='same',
                  kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                  kernel_initializer='he_normal',
                  kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(y)
    y = BatchNormalization()(y)
  
    # Overall residual, connect weight layer and identity paths
    x = Add()([x, y]) 
    x = QActivation(activation=quantized_relu(activation_total_bits, activation_int_bits))(x)

    if len(num_filters) > 2 and num_filters[2] > 0 and strides[1] != '' and kernel_sizes[3] > 0:
        # Second stack
        # Weight layers
        y = QConv2D(num_filters[2],
                    kernel_size=kernel_sizes[3],
                    strides=int(strides[1][0]),
                    padding='same',
                    kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                    bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)
        y = BatchNormalization()(y)
        y = QActivation(activation=quantized_relu(activation_total_bits, activation_int_bits))(y)
        y = QConv2D(num_filters[3],
                    kernel_size=kernel_sizes[4],
                    strides=int(strides[1][1]),
                    padding='same',
                    kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                    bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(y)
        y = BatchNormalization()(y)
        
        # Adjust for change in dimension due to stride in identity
        x = QConv2D(num_filters[3],
                    kernel_size=kernel_sizes[5],
                    strides=int(strides[1][2]),
                    padding='same',
                    kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                    bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)

        # Overall residual, connect weight layer and identity paths
        x = Add()([x, y])
        x = QActivation(activation=quantized_relu(activation_total_bits, activation_int_bits))(x)

    if len(num_filters) > 4 and num_filters[4] > 0 and strides[2] != '' and kernel_sizes[6] > 0:
        # Third stack
        # Weight layers
        y = QConv2D(num_filters[4],
                    kernel_size=kernel_sizes[6],
                    strides=int(strides[2][0]),
                    padding='same',
                    kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                    bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)
        y = BatchNormalization()(y)
        y = QActivation(activation=quantized_relu(activation_total_bits, activation_int_bits))(y)
        y = QConv2D(num_filters[5],
                    kernel_size=kernel_sizes[7],
                    strides=int(strides[2][1]),
                    padding='same',
                    kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                    bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(y)
        y = BatchNormalization()(y)
  
        # Adjust for change in dimension due to stride in identity
        x = QConv2D(num_filters[5],
                    kernel_size=kernel_sizes[8],
                    strides=int(strides[2][2]),
                    padding='same',
                    kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                    bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                    kernel_initializer='he_normal',
                    kernel_regularizer=l1_l2(l1=l1p,l2=l2p))(x)
        
        # Overall residual, connect weight layer and identity paths
        x = Add()([x, y])
        x = QActivation(activation=quantized_relu(activation_total_bits, activation_int_bits))(x)

    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    # Do we want Qlayer of this below?
    x = QAveragePooling2D(pool_size=pool_size, quantizer=quantized_relu(activation_total_bits, activation_int_bits))(x)
    y = Flatten()(x)
    # Changed output to separate QDense but did not quantize softmax as specified, is this the way you wanted it?
    outputs = QDense(num_classes,
                     kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                     bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
                     kernel_initializer='he_normal')(y)
    outputs = Activation('softmax', name='softmax')(outputs)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
