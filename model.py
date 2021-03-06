import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.models import model_from_json

import cv2
import numpy as np
from matplotlib import pyplot as plt

class Model(object):
	def __init__(self):
		super(Model, self).__init__()
		self.model = None

	def build_model(self):

		self.model = Sequential()
		self.model.add(Conv2D(64, kernel_size=(9, 9),
		                 activation='relu',
		                 padding = 'same',
		                 input_shape=(32,32,3)))
		self.model.add(Conv2D(32, (1, 1), activation='relu',padding = 'same'))
		self.model.add(Conv2D(3, (5, 5), activation='relu',padding = 'same'))

		self.model.compile(loss=keras.losses.mean_squared_error,
		              optimizer=keras.optimizers.Adadelta(),
		              metrics=['accuracy'])

	def load_model(self, model_name, weights_name):

		json_file = open(model_name, 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(loaded_model_json)
		#	 load weights into new model
		self.model.load_weights(weights_name)
		print("Loaded model from disk")

	def save_model(self, model_name, weights_name):

		model_json = self.model.to_json()
		with open(model_name, "w") as json_file:
		    json_file.write(model_json)
		# serialize weights to HDF5
		self.model.save_weights(weights_name)
		print("Saved model to disk")

	def compile_model(self):
		self.model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

	def train_model(self,epochs,batch_size,X_train,X_dev,Y_train,Y_dev):

		self.model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_dev, Y_dev))



	def predict_deblur(self,img_path):
		img = cv2.imread(img_path)
		im2 = img.copy()
		im2[:, :, 0] = img[:, :, 2]
		im2[:, :, 2] = img[:, :, 0]
		plt.imshow(im2)
		plt.figure()

		img = np.asarray(img)
		img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
		img = img.astype('float32')
		img /= 255
		pred_img = img
		for x in range(int(img.shape[1]/20)-1):
		    for y in range(int(img.shape[2]/20)-1):
		        pred = self.model.predict(img[:,x*20:x*20 +32,y*20:y*20 + 32,:])
		        pred_img[:,x*20 + 6:(x+1)*20 + 6,y*20 + 6:(y+1)*20 + 6,:] = pred[:,6:26,6:26,:]
		pred_img *=255
		pred_img = pred_img.astype('int32')
		im3 = pred_img[0].copy()
		im3[:, :, 0] = pred_img[0][:, :, 2]
		im3[:, :, 2] = pred_img[0][:, :, 0]
		plt.imshow(im3)
		plt.figure()
		cv2.imwrite('clear_'+img_path,pred_img[0])



