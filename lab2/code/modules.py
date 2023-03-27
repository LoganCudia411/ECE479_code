# -*- coding: utf-8 -*-
## for ECE479 ICC Lab2 Part3

'''
*modules files*
'''
# *************** DO NOT TOUCH THIS FILE! ***************

import tensorflow as tf
# Includes all keras modules required
from functools import partial
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import add
from tensorflow.keras import backend as K


#######################################################


# Auxiliary functions definition
def scaling(x, scale):
    # Used for values scaling by element-wise multiplication
    return x * scale


def generate_layer_name(name, branch_idx=None, prefix=None):
    # Used for generating name for corresponding layers
    if prefix is None:
        return None
    if branch_idx is None:
        return '_'.join((prefix, name))
    return '_'.join((prefix, 'Branch', str(branch_idx), name))
###############################################################
