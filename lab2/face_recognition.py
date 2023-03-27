import cv2
import numpy as np
def pre_process(face,required_size=(160,160)):
	ret = cv2.resize(face,required_size)
	ret = ret.astype('float32')
	mean, std = ret.mean(), ret.std()
	ret = (ret-mean)/std
	return ret
image = cv2.imread("quan_cropped.jpg")
img_arr = np.array(image)
pre_processed_im =pre_process(face=img_arr)
print(pre_processed_im.shape)
cv2.imwrite('preprocess2_quan.jpg',pre_processed_im)

