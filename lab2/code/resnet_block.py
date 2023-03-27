# -*- coding: utf-8 -*-
## for ECE479 ICC Lab2 Part3

'''
*Definition for reusable resnet block*
'''

from conv2d_bn import *


def resnet_block(x, scale, block_idx, block_type,
                           activation='relu'):
    # "channel_axis" refers to image channels, channel axis will remained untouched during the convolution process
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3

    # Do Not Touch!  Used for generating layer name and aim for easy managing of the structure
    if block_idx is None:
        prefix = None
    else:
        prefix = '_'.join((block_type, str(block_idx)))
    name_fmt = partial(generate_layer_name, prefix=prefix)
    #######################################################################################

    # Main definition for the reusable resnet block starts here
    if block_type == 'Inception_block_c':
        branch_0 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_0_1x1', 0))
        branch_1 = conv2d_bn(x, 192, 1, name=name_fmt('Conv2d_1a_1x1', 1))
        branch_1 = conv2d_bn(branch_1,
                             192, [1, 3],
                             name=name_fmt('Conv2d_1b_1x3', 1))
        branch_1 = conv2d_bn(branch_1,
                             192, [3, 1],
                             name=name_fmt('Conv2d_1c_3x1', 1))
        branches = [branch_0, branch_1]


    elif block_type == 'Inception_block_a':
        branch_0 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_0_1x1', 0))
        branch_1 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_1a_1x1', 1))
        branch_1 = conv2d_bn(branch_1,
                             32,
                             3,
                             name=name_fmt('Conv2d_1b_3x3', 1))
        branch_2 = conv2d_bn(x, 32, 1, name=name_fmt('Conv2d_2a_1x1', 2))
        branch_2 = conv2d_bn(branch_2,
                             32,
                             3,
                             name=name_fmt('Conv2d_2b_3x3', 2))
        branch_2 = conv2d_bn(branch_2,
                             32,
                             3,
                             name=name_fmt('Conv2d_2c_3x3', 2))
        branches = [branch_0, branch_1, branch_2]


    elif block_type == 'Inception_block_b':
        branch_0 = conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_0_1x1', 0))
        branch_1 = conv2d_bn(x, 128, 1, name=name_fmt('Conv2d_1a_1x1', 1))
        branch_1 = conv2d_bn(branch_1,
                             128, [1, 7],
                             name=name_fmt('Conv2d_1b_1x7', 1))
        branch_1 = conv2d_bn(branch_1,
                             128, [7, 1],
                             name=name_fmt('Conv2d_1c_7x1', 1))
        branches = [branch_0, branch_1]

    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "Inception_block_a", "Inception_block_b" or "Inception_block_c", '
                         'but got: ' + str(block_type))

    # Concatenate all branches and connect the block to next node
    mixed = Concatenate(axis=channel_axis,
                        name=name_fmt('Concatenate'))(branches)


    # Do Not Touch
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=name_fmt('Conv2d_1x1'))
    up = Lambda(scaling,
                output_shape=K.int_shape(up)[1:],
                arguments={'scale': scale})(up)
    x = add([x, up])
    ############################################################

    # Add a final "activation" layer
    if activation is not None:
        x = Activation(activation, name=name_fmt('Activation'))(x)

    return x
