import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
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
    fig.savefig('cnn1d_test.png')
    plt.show()

class Cnn1D:

	def __init__(self, classes):
		self.CLASSES = classes

	def define_cnn_model(self):

		pickle_in = open("X_train_mfcc_42000.pickle","rb")
		X_train = pickle.load(pickle_in)

		pickle_in = open("y_train_mfcc_42000.pickle","rb")
		y_train = pickle.load(pickle_in)		
		
		#converting to array
		X_train = np.asarray(X_train)
		print(X_train.shape)
		X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)

		#model selection
		data = Input(shape = (32,1))
		x = Conv1D(100, 10, padding = 'same', activation = 'relu', kernel_regularizer = l2(0.001))(data)
		x = Dropout(.25)(x)
		x = MaxPooling1D(2)(x)

		x = Conv1D(100, 10, padding = 'same', activation = 'relu', kernel_regularizer = l2(0.001))(x)
		x = Dropout(.25)(x)
		x = MaxPooling1D(2)(x)

		x = Conv1D(100, 5, activation = 'relu', kernel_regularizer = l2(0.001))(x)
		x = Dropout(.25)(x)

		x = Conv1D(100, 5, activation = 'relu', kernel_regularizer = l2(0.001))(x)
		x = Dropout(.25)(x)

		x = Flatten()(x)

		x = Dense(256, activation = 'relu')(x)
		x = Dropout(.50)(x)

		x = Dense(len(CLASSES), activation = 'softmax')(x)
		
		model = Model(data, x)

		return model, X_train, y_train

class Cnn2D:

	def define_cnn_model(self):
		
		pickle_in = open("X_train_mfcc_32000.pickle","rb")
		X_train = pickle.load(pickle_in)

		pickle_in = open("y_train_mfcc_32000.pickle","rb")
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

	CLASSES = ['Old_Talker_Angry', 'Old_Talker_Disgust', 'Old_Talker_Fear', 'Old_Talker_Happy', 'Old_Talker_Neutral', 'Old_Talker_Pleasent_Surprise', 'Old_Talker_Sad']

	cnn = Cnn1D(CLASSES)
	# cnn = Cnn2D(CLASSES)

	model, X_train, y_train = cnn.define_cnn_model()

	init_lr = 0.0001
	total_epochs = 100
	opt = Adam(lr = init_lr, decay = init_lr / total_epochs)
	model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

	model.summary()

	model_info = model.fit(X_train, y_train, batch_size = 20, epochs = total_epochs, validation_split = .2)
	plot_model_history(model_info)
	model_json = model.to_json()
	with open("model_mfcc_cnn1d_test.json", "w") as json_file:
		json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights("model_mfcc_cnn1d_test.h5")
		print("Saved model to disk")