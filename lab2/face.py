from mtcnn import MTCNN
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
from picamera2 import Picamera2, Preview
from matplotlib.patches import Rectangle

import time
def capture_image_func():
	#picam2 = Picamera2()
	#picam2.start_and_capture_file("im.jpg")
	image = cv2.imread("ke_huy")
	#image = cv2.imread("im.jpg")
	image = image[:,:,::-1]
	return image
def detect_and_crop(mtcnn,image):
	detection = mtcnn.detect_faces(image)
	print(type(detection))
	if len(detection) > 0:
		x,y,w,h = detection[0]['box']
		x_new=int(x*0.9)
		y_new=int(y*0.9)
		x_ext = int((w+x)*1.1)
		y_ext = int((h+y)*1.1)
		cropped_image = image[y_new:y_ext,x_new:x_ext]
		cv2.imwrite('quan_cropped.jpg',cropped_image)
		return cropped_image,x,y,w,h
	else:
		return None, None
mtcnn = MTCNN()
im = capture_image_func()
cropped_image,bound_x,bound_y,bound_w,bound_h = detect_and_crop(mtcnn,im)

bounding_box = [bound_x,bound_y,bound_w,bound_h]
def show_bounding_box(image, bounding_box):
	x1, y1, w, h = bounding_box
	fig, ax = plt.subplots(1,1)
	ax.imshow(image)
	ax.add_patch(Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none'))
	plt.show()
	return
show_bounding_box(im,bounding_box)

#plt.imshow(im)
#plt.show()
#print(type(im))
print("Ran without Error")
