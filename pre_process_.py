import cv2
import os
from keras.preprocessing.image import load_img
import numpy as np
from keras.preprocessing import image

files = os.listdir(os.getcwd() + '/data/')
data_img_and_label = []
for f in files:
	img = image.load_img(os.getcwd() + '/data/' + f, target_size=(100,100),grayscale=True)
	img = np.array(img)
	temp_label = f.split('_')
	data_img_and_label.append((img,temp_label[0]))

print (data_img_and_label)
data_img_and_label = np.array(data_img_and_label)
np.save('NN_data.npy',data_img_and_label)
