import numpy as np

from keras.layers import Input,Dense,Flatten,Dropout,Activation
from keras.applications import VGG16,Xception,ResNet50
from keras.layers import BatchNormalization
from keras import Model,models,Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD


class ImageClassifier():
	def __init__(self):

		# self.model = build_model()
		self.batch_size = 64
		self.image_shape = (64,64,3)
		self.epochs = 20
		self.num_classes = 18
		self.src_dir = 'data/train/'


		self.model = self.build_model()
		self.img_gen = self.image_generator()
		self.build()


	def build_model(self):
	
		input_img = Input(shape=self.image_shape)
		x = BatchNormalization()(input_img)
	
		x = VGG16(include_top=False,input_shape=self.image_shape)(x)

		x = BatchNormalization()(x)
		x = Dense(2048,activation='relu')(x)
		x = Dropout(0.5)(x)
		x = BatchNormalization()(x)
		x = Dense(2048,activation='relu')(x)

		x = Dense(self.num_classes)(x)
		prediction = Activation('softmax')(x)

		model = Model(input_img,prediction)
		return model


	def image_generator(self):
		image_gen = ImageDataGenerator(
			horizontal_flip=True,
			rotation_range = 0.1,
			width_shift_range = 0.2,
			rescale=1/255.0,
			validation_split=0.1,
			)

		image_gen = image_gen.flow_from_directory(
			self.src_dir,
			target_size=(150, 150),
			batch_size=2048,
			class_mode='categorical')

		return image_gen

	def build(self):

		opt = SGD(0.01,0.8,nesterov=True)

		self.model.compile(
			optimizer = opt,
			metrics=['accuracy'],
			loss = 'categorical_crossentropy'
			)


	def train(self):


		self.model.summary()

		self.model.fit_generator(
			self.image_gen,
			use_multiprocessing = True
			)

	def validation(self):
		pass

	def evaluate(self):
		pass


clf = ImageClassifier()
clf.train()
