## for ECE479 ICC Lab2 Part3

'''
*Main Student Script*
'''

# Your works start here

# Import packages you need here
from inception_resnet import InceptionResNetV1Norm
import numpy as np
import tensorflow as tf

# Create a model
model = InceptionResNetV1Norm()

# Verify the model and load the weights into the net
print(model.summary())
print(len(model.layers))
model.load_weights("./weights/inception_keras_weights.h5")  # Has been translated from checkpoint
model.save("google_net")
converter = tf.lite.TFLiteConverter.from_saved_model("/Users/lorenzo/Documents/UIUC/Y2-S2/ECE479/lab2_sp23/code/google_net")
tflite_model = converter.convert()
open("google_net.tflite","wb").write(tflite_model)