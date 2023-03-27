# -*- coding: utf-8 -*-
## for ECE479 ICC Lab2 Part3

'''
*Keras definition for inception resnet v1*
'''

# Import all packages
from resnet_block import *

# Definition for the Inception ResNet Structure starts here
def InceptionResNetV1Norm(input_shape=(160, 160, 3),
                          classes=512,
                          dropout_keep_prob=0.8,
                          weights_path=None):

    # DO NOT TOUCH
    inputs = Input(shape=input_shape)
    # Example of how to use conv2d_bn
    # Note: this is also part of the netowrk, do not delete it
    x = conv2d_bn(inputs,
                  32,
                  3,
                  strides=2,
                  padding='valid',
                  name='Conv2d_1a_3x3')
    ############################################

    # Preprocess inputs by MaxPooling2D
    ## TO DO Step 1 : Finish the implementation for preprocessing with given parameters
    # Please name all layers properly to make it easy for your debugging
    # Your code goes here
    x = conv2d_bn(x,32,(3,3),padding='valid',name='first_conv')
    x = conv2d_bn(x,64,(3,3),name='second_conv')
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(x)
    x = conv2d_bn(x,80,(1,1),padding='valid',name='third_conv')
    x = conv2d_bn(x,192,(3,3),padding='valid',name='fourth_conv')
    x = conv2d_bn(x,256,(3,3),strides=3,padding='valid')

    




    ##############################################

    # 5x Inception-ResNet-A block:
    ## TO DO Step 2 : Finish the implementation for Inception-A block with given parameters
    # Please name all blocks properly to make it easy for your debugging
    # Hint : Use for loop to instantiate multiples reception blocks
    # Your code goes here
    for block_idx in range(5):
        x = resnet_block(x,scale=0.17,block_idx=block_idx, block_type='Inception_block_a',activation='relu')
    




    ###############################################

    # Mixed 6a (Reduction-A block):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    name_fmt = partial(generate_layer_name, prefix='Mixed_6a')
    branch_0 = conv2d_bn(x,
                         384,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 0))
    branch_1 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1, 192, 3, name=name_fmt('Conv2d_0b_3x3', 1))
    branch_1 = conv2d_bn(branch_1,
                         256,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 1))
    branch_pool = MaxPooling2D(3,
                               strides=2,
                               padding='valid',
                               name=name_fmt('MaxPool_1a_3x3', 2))(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_6a')(branches)

    # 10x Inception-ResNet-B block:
    ## TO DO Step 3 : Finish the implementation for Inception-B block with given parameters
    # Please name all blocks properly to make it easy for your debugging
    # Hint : Use for loop to instantiate multiples reception blocks
    # Your code goes here
    for block_idx in range(10):
        x = resnet_block(x,scale=0.1,block_idx=block_idx,block_type='Inception_block_b',activation='relu')




    ###############################################

    # Mixed 7a (Reduction-B block)
    name_fmt = partial(generate_layer_name, prefix='Mixed_7a')
    branch_0 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 0))
    branch_0 = conv2d_bn(branch_0,
                         384,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 0))
    branch_1 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 1))
    branch_1 = conv2d_bn(branch_1,
                         256,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 1))
    branch_2 = conv2d_bn(x, 256, 1, name=name_fmt('Conv2d_0a_1x1', 2))
    branch_2 = conv2d_bn(branch_2, 256, 3, name=name_fmt('Conv2d_0b_3x3', 2))
    branch_2 = conv2d_bn(branch_2,
                         256,
                         3,
                         strides=2,
                         padding='valid',
                         name=name_fmt('Conv2d_1a_3x3', 2))
    branch_pool = MaxPooling2D(3,
                               strides=2,
                               padding='valid',
                               name=name_fmt('MaxPool_1a_3x3', 3))(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='Mixed_7a')(branches)

    # 5x Inception-ResNet-C block:
    ## TO DO Step 4 : Finish the implementation for Inception-B block with given parameters
    # Please name all blocks properly to make it easy for your debugging
    # Hint : Use for loop to instantiate multiples reception blocks
    # Your code goes here
    for block_idx in range(5):
        x = resnet_block(x,scale=0.2,block_idx=block_idx, block_type='Inception_block_c',activation='relu')




    ###############################################

    # Final Inception Block
    x = resnet_block(x,
                     scale=1.,
                     block_idx=6,
                     block_type='Inception_block_c',
                     activation=None
                               )


    # Classification block
    ## TO DO Step 5 : Apply Global Average pooling + Dropout layers
    # Please name all blocks properly to make it easy for your debugging
    x = tf.keras.layers.GlobalAveragePooling2D()(x)# Your code goes here (do not modify the variable name assigned to)
    x = tf.keras.layers.Dropout(dropout_keep_prob)(x)# Your code goes here (do not modify the variable name assigned to)


    ## DO NOT TOUCH
    # (BottleNeck blcok with Normalization)
    x = Dense(classes, use_bias=False, name='Bottleneck')(x)
    bn_name = generate_layer_name('BatchNorm', prefix='Bottleneck')
    x = BatchNormalization(momentum=0.995,
                           epsilon=0.001,
                           scale=False,
                           name=bn_name)(x)
    x = Lambda(K.l2_normalize, arguments={'axis': 1}, name='normalize')(x)

    # Create model
    model = Model(inputs, x, name='inception_resnet_v1')
    if weights_path is not None:
        model.load_weights(weights_path)

    return model
    #############################################################