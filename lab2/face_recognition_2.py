import time
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
#def run_model(model,face):
def run_model(model,face):
	interpreter = tflite.Interpreter(model_path=model)
	
	interpreter.allocate_tensors()
	
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	
	input_shape = input_details[0]['shape']
	
	interpreter.set_tensor(input_details[0]['index'],face)
	interpreter.invoke()
	output_data = interpreter.get_tensor(output_details[0]['index'])
	predicted_class = np.argmax(output_data)
	
	return output_data
image = cv2.imread("preprocess2_quan.jpg")
image_reshaped = np.expand_dims(image,axis=0)

image1_out = run_model("/home/siliconsquad/Desktop/google_net_new2.tflite",image_reshaped.astype(np.float32))

image2 = cv2.imread("preprocess2_junhao.jpg")
image2_reshaped = np.expand_dims(image2,axis=0)
image2_out = run_model("/home/siliconsquad/Desktop/google_net_new2.tflite",image2_reshaped.astype(np.float32))

print(np.sqrt(np.sum((image1_out-image2_out)**2,axis=1)))

