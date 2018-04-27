import numpy as np
import pandas as pd

from keras.layers import Input,Dense,Flatten,Dropout,Activation
from keras.layers import BatchNormalization,Reshape
from keras.applications import VGG16,Xception,ResNet50
from keras import Model,models,Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD,RMSprop
from keras.models import save_model,load_model
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.callbacks import TensorBoard

# np.random.seed(32)

class ImageClassifier():
	def __init__(self):

		self.name = 'Xception'
		self.length = 139
		self.breadth = 139
		self.image_shape = (self.length,self.breadth,3)
		
		self.batch_size = 50
		self.epochs = 50
		self.num_classes = 18
		
		self.train_folder = 'data/fold1/train/'
		self.validation_folder = 'data/fold1/validation/'
		self.test_folder = 'data/test/'
		self.save_path = 'models/'+self.name+'_best.h5'

		self.model = self.build_model()

		self.build()


	def get_basemodel(self):

		base_model = Xception(weights = "imagenet", include_top=False, input_shape = self.image_shape)
		for layer in base_model.layers:
			layer.trainable = True
		return base_model

	def build_model(self):
		
		base_model = self.get_basemodel()
		x = base_model.output
		x = Flatten()(x)
		x = Dense(512,activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(512,activation='relu')(x)
		x = Dropout(0.5)(x)
		predictions = Dense(self.num_classes,activation='softmax')(x)
		model = Model(input=base_model.input,output=predictions)
		return model

	def train_generator(self):
		image_gen = ImageDataGenerator(
			horizontal_flip=True,
			rotation_range = 0.2,
			zoom_range = 0.2,
			width_shift_range = 0.2,
			height_shift_range = 0.2,
			rescale=1/255.0,
			fill_mode = 'nearest'
			)

		image_gen = image_gen.flow_from_directory(
			self.train_folder,
			target_size=(self.length, self.breadth),
			batch_size=self.batch_size,
			class_mode='categorical')

		return image_gen

	def validation_generator(self):
		image_gen = ImageDataGenerator(
			rescale=1/255.0,
			)

		image_gen = image_gen.flow_from_directory(
			self.validation_folder,
			target_size=(self.length, self.breadth),
			batch_size=self.batch_size,
			class_mode='categorical')

		return image_gen

	def test_generator(self):
		image_gen = ImageDataGenerator(
			rescale=1/255.0,
			)

		image_gen = image_gen.flow_from_directory(
			self.test_folder,
			target_size=(self.length, self.breadth),
			batch_size=self.batch_size,
			class_mode=None)

		return image_gen

	def get_callbacks(self):
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, verbose=1, mode='auto')
		checkpointer = ModelCheckpoint(filepath=self.save_path, verbose=1, save_best_only=True)
		tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=self.batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
		return [early_stopping,checkpointer,tensorboard]
		# return [checkpointer]

	def build(self,lr=0.01):

		# opt = RMSprop(lr=lr, decay=1e-6)
		# opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
		opt = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

		self.model.compile(
			optimizer = opt,
			metrics=['accuracy'],
			loss = 'categorical_crossentropy'
			)

		self.model.summary()
		return self.model


	def train(self):

		hist = self.model.fit_generator(
			generator = self.train_generator(),
			validation_data	= self.validation_generator(),
			epochs = self.epochs,
			callbacks = self.get_callbacks(),
			)

	def continue_training(self,lr=0.01):

		self.model = load_model(self.save_path)
		self.model = self.build(lr=lr)

		hist = self.model.fit_generator(
			generator = self.train_generator(),
			validation_data	= self.validation_generator(),
			epochs = self.epochs,
			callbacks = self.get_callbacks(),
			)


	def evaluate(self):
		output = self.model.predict(
			generator = self.test_generator(),
			batch_size = self.batch_size
			)

		print output
		output =  np.argmax(output,axis = 1)
		print output

		return output


if __name__=='__main__':
	clf = ImageClassifier()
	# clf.train()
	clf.continue_training()
