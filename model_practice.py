import os
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
from feature import Feature

class Cnn1D:
	def define_model(self, working_directory, training_directory, happy_path, sad_path, confused_path, neutral_path):

		#necessary directories for feature class
		self.working_directory = working_directory
		self.training_directory = training_directory
		self.happy_path = happy_path
		self.sad_path  = sad_path
		self.confused_path = confused_path
		self.neutral_path = neutral_path

		#feture extraction
		feature = Feature(self.working_directory, self.training_directory, self.happy_path, self.sad_path, self.confused_path, self.neutral_path)
		CLASSES = ['happy', 'sad']
		emotion_feature_label = feature.feature_extraction(CLASSES, 16000, 'kaiser_best')
		X_train, y_train = feature.feature_label_separator(emotion_feature_label, 16000)

		#converting to array
		#X_train = np.asarray(X_train)

		#model selection
		model = Sequential()

		model.add(Flatten(data_format = channels_last))

		model.add(Dense(256))
		model.add(Activation('relu'))

		model.add(Dense(256))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))

		model.add(Dense(CLASSES))
		model.add(Activation('softmax'))

		return model, X_train, y_train

if __name__ == '__main__':

	#defining directories
	working_directory = os.getcwd()
	training_directory = os.path.join(working_directory, 'train')
	happy_path = os.path.join(training_directory, 'happy')
	sad_path = os.path.join(training_directory, 'sad')
	confused_path = os.path.join(training_directory, 'confused')
	neutral_path = os.path.join(training_directory, 'neutral')

	cnn = Cnn1D()

	model, X_train, y_train = cnn.define_model(working_directory, training_directory, happy_path, sad_path, confused_path, neutral_path)

	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

	model.fit(X_train, y_train, batch_size = 32, epochs = 5, validation_split = .2)