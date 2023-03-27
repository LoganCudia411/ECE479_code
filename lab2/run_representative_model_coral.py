import time
import tflite_runtime.interpreter as tflite
import numpy as np

interpreter = tflite.Interpreter(model_path="representative_converted_model.tflite",experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
start = time.time()

test_images = np.load("test_images.npy")
test_images*=255.0
test_labels = np.load("test_labels.npy")
counter =0

for i in range(10000):
	test_images_reshaped = np.expand_dims(test_images,axis=3)
	test_images_reshaped = test_images_reshaped[i:i+1]
	
	
	test_images_reshaped_float32 = test_images_reshaped.astype('uint8')
	interpreter.set_tensor(input_details[0]['index'],test_images_reshaped_float32)
	interpreter.invoke()
	output_data = interpreter.get_tensor(output_details[0]['index'])
	predicted_class = np.argmax(output_data)
	if(predicted_class==test_labels[i]):
		counter+=1
end=time.time()
print(f'{counter/10000*100}% Accurate\nTime Taken:{end-start}')

	
