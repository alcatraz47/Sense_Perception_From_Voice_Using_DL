import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.regularizers import l2
from sklearn import metrics
from feature import Feature

import matplotlib as mpl
mpl.use('TkAgg')

def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('cnn2d.png')
    plt.show()

class Cnn1D:

	def define_cnn_model(self, working_directory, training_directory, old_Talker_Angry, old_Talker_Disgust, old_Talker_Fear, old_Talker_Happy, old_Talker_Neutral, old_Talker_Pleasent_Surprise, old_Talker_Sad):

		#necessary directories for feature class
		self.working_directory = working_directory
		self.training_directory = training_directory
		self.old_Talker_Angry = old_Talker_Angry
		self.old_Talker_Disgust = old_Talker_Disgust
		self.old_Talker_Fear = old_Talker_Fear
		self.old_Talker_Happy = old_Talker_Happy
		self.old_Talker_Neutral = old_Talker_Neutral
		self.old_Talker_Pleasent_Surprise = old_Talker_Pleasent_Surprise
		self.old_Talker_Sad = old_Talker_Sad
        # self.young_Talker_Angry = young_Talker_Angry
        # self.young_Talker_Fear = young_Talker_Fear
        # self.young_Talker_Neutral = young_Talker_Neutral
        # self.young_Talker_Pleasant_Surprise = young_Talker_Pleasant_Surprise
        # self.young_Talker_Sad = young_Talker_Sad
        # self.young_Walker_Disgust = young_Walker_Disgust
        # self.younger_Talker_Happy = younger_Talker_Happy

		#feture extraction
		CLASSES = ['Old_Talker_Angry', 'Old_Talker_Disgust', 'Old_Talker_Fear', 'Old_Talker_Happy', 'Old_Talker_Neutral', 'Old_Talker_Pleasent_Surprise', 'Old_Talker_Sad']
		
		pickle_in = open("X.pickle","rb")
		X_train = pickle.load(pickle_in)

		pickle_in = open("y.pickle","rb")
		y_train = pickle.load(pickle_in)		
		
		#converting to array
		X_train = np.asarray(X_train)
		print(X_train.shape)
		X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)

		#model selection
		model = Sequential()

		model.add(Conv1D(100, 10, padding = 'same', input_shape = (32,1), kernel_regularizer = l2(0.0001)))
		model.add(Activation('relu'))

		model.add(Conv1D(100, 10, padding = 'same', kernel_regularizer = l2(0.0001)))
		model.add(Activation('relu'))
		model.add(MaxPool1D(2))

		model.add(Conv1D(160, 5, kernel_regularizer = l2(0.0001)))
		model.add(Activation('relu'))

		model.add(Conv1D(160, 5, kernel_regularizer = l2(0.0001)))
		model.add(Activation('relu'))

		model.add(Flatten())

		model.add(Dense(256))
		model.add(Activation('relu'))

		#model.add(Dropout(0.5))

		model.add(Dense(len(CLASSES)))
		model.add(Activation('softmax'))

		return model, X_train, y_train

class Cnn2D:

	def define_cnn_model(self, working_directory, training_directory, old_Talker_Angry, old_Talker_Disgust, old_Talker_Fear, old_Talker_Happy, old_Talker_Neutral, old_Talker_Pleasent_Surprise, old_Talker_Sad):

		#necessary directories for feature class
		self.working_directory = working_directory
		self.training_directory = training_directory
		self.old_Talker_Angry = old_Talker_Angry
		self.old_Talker_Disgust = old_Talker_Disgust
		self.old_Talker_Fear = old_Talker_Fear
		self.old_Talker_Happy = old_Talker_Happy
		self.old_Talker_Neutral = old_Talker_Neutral
		self.old_Talker_Pleasent_Surprise = old_Talker_Pleasent_Surprise
		self.old_Talker_Sad = old_Talker_Sad
        # self.young_Talker_Angry = young_Talker_Angry
        # self.young_Talker_Fear = young_Talker_Fear
        # self.young_Talker_Neutral = young_Talker_Neutral
        # self.young_Talker_Pleasant_Surprise = young_Talker_Pleasant_Surprise
        # self.young_Talker_Sad = young_Talker_Sad
        # self.young_Walker_Disgust = young_Walker_Disgust
        # self.younger_Talker_Happy = younger_Talker_Happy

		#feture extraction
		CLASSES = ['Old_Talker_Angry', 'Old_Talker_Disgust', 'Old_Talker_Fear', 'Old_Talker_Happy', 'Old_Talker_Neutral', 'Old_Talker_Pleasent_Surprise', 'Old_Talker_Sad']
		
		pickle_in = open("X.pickle","rb")
		X_train = pickle.load(pickle_in)

		pickle_in = open("y.pickle","rb")
		y_train = pickle.load(pickle_in)		
		
		#converting to array
		X_train = np.asarray(X_train)
		X_train = X_train.reshape(-1, X_train.shape[0], X_train.shape[1], 1)
		print(X_train.shape[0])
		print(y_train.shape)

		#model selection
		model = Sequential()
		model.add(Conv2D(16, (3,3), activation = 'relu', strides=(1,1), padding ='same', input_shape = (X_train.shape[1], X_train.shape[2], 1)))
		model.add(Conv2D(32, (3,3), activation = 'relu', strides=(1,1), padding ='same'))
		model.add(Conv2D(64, (3,3), activation = 'relu', strides=(1,1), padding ='same'))
		model.add(Conv2D(128, (3,3), activation = 'relu', strides=(1,1), padding ='same'))
		
		model.add(MaxPool2D((2,2)))
		model.add(Dropout(0.5))
		model.add(Flatten())
		model.add(Dense(128, activation = 'relu'))
		model.add(Dense(64, activation = 'relu'))
		model.add(Dense(len(CLASSES), activation = 'softmax'))

		return model, X_train, y_train

if __name__ == '__main__':

	#defining directories
	working_directory = os.getcwd()
	training_directory = os.path.join(working_directory, 'Toronto_Dataset')
	old_Talker_Angry = os.path.join(training_directory, 'Old_Talker_Angry')
	old_Talker_Disgust = os.path.join(training_directory, 'Old_Talker_Disgust')
	old_Talker_Fear = os.path.join(training_directory, 'Old_Talker_Fear')
	old_Talker_Happy = os.path.join(training_directory, 'Old_Talker_Happy')
	old_Talker_Neutral = os.path.join(training_directory, 'Old_Talker_Neutral')
	old_Talker_Pleasent_Surprise = os.path.join(training_directory, 'Old_Talker_Pleasent_Surprise')
	old_Talker_Sad = os.path.join(training_directory, 'Old_Talker_Sad')

	#cnn = Cnn1D()
	cnn = Cnn2D()

	model, X_train, y_train = cnn.define_cnn_model(working_directory, training_directory, old_Talker_Angry, old_Talker_Disgust, old_Talker_Fear, old_Talker_Happy, old_Talker_Neutral, old_Talker_Pleasent_Surprise, old_Talker_Sad)

	init_lr = 0.0001
	total_epochs = 100
	opt = Adam(lr = init_lr, decay = init_lr / total_epochs)
	model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

	model_info = model.fit(X_train, y_train, batch_size = 20, epochs = total_epochs, validation_split = .2)
	plot_model_history(model_info)
	model_json = model.to_json()
	with open("model_cnn2d_old_talker.json", "w") as json_file:
		json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights("model_cnn2d_old_talker.h5")
		print("Saved model to disk")