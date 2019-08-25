import os
import numpy as np
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics
from feature import Feature
from keras.layers import Embedding, SimpleRNN
import pickle
import matplotlib.pyplot as plt
from keras.regularizers import l2
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
	fig.savefig('rnn_lstm.png')
	plt.show()

class Rnn:
	def define_rnn_model(self, working_directory, training_directory, old_Talker_Angry, old_Talker_Disgust, old_Talker_Fear, old_Talker_Happy, old_Talker_Neutral, old_Talker_Pleasent_Surprise, old_Talker_Sad):
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

		CLASSES = ['Old_Talker_Angry', 'Old_Talker_Disgust', 'Old_Talker_Fear', 'Old_Talker_Happy', 'Old_Talker_Neutral', 'Old_Talker_Pleasent_Surprise', 'Old_Talker_Sad']
		#converting to array
		pickle_in = open("X.pickle","rb")
		X_train = pickle.load(pickle_in)

		pickle_in = open("y.pickle","rb")
		y_train = pickle.load(pickle_in)		

		#converting to array
		X_train = np.asarray(X_train)
		print(X_train.shape[1])
		# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)

		model = Sequential()
		model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
		model.add(LSTM(100, return_sequences=True))
		model.add(LSTM(100, return_sequences=True))
		model.add(LSTM(100, return_sequences=True))
		model.add(LSTM(100, return_sequences=True))
		model.add(LSTM(100, return_sequences=False))
		model.add(Dropout(0.2))
		model.add(Dense(len(CLASSES), activation = "softmax"))

		# model = Sequential()
		# model.add(Embedding(10000, 32))
		# model.add(SimpleRNN(32, return_sequences=True, kernel_regularizer = l2(0.0001)))
		# model.add(SimpleRNN(32, return_sequences=True, kernel_regularizer = l2(0.0001)))
		# model.add(SimpleRNN(32, return_sequences=True, kernel_regularizer = l2(0.0001)))
		# model.add(SimpleRNN(32, return_sequences=True))
		# model.add(SimpleRNN(32, return_sequences=True))
		# model.add(SimpleRNN(32, return_sequences=True))
		# model.add(SimpleRNN(32))
		# model.add(Dense(len(CLASSES), activation='softmax'))
		model.summary()

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

	rnn = Rnn()

	model, X_train, y_train = rnn.define_rnn_model(working_directory, training_directory, old_Talker_Angry, old_Talker_Disgust, old_Talker_Fear, old_Talker_Happy, old_Talker_Neutral, old_Talker_Pleasent_Surprise, old_Talker_Sad)

	init_lr = 0.0001
	total_epochs = 100
	opt = Adam(lr = init_lr, decay = init_lr / total_epochs)
	model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
	X_train = np.array(X_train).reshape(1400, 32, 1)
	print(X_train.shape)

	model_info = model.fit(X_train, y_train, batch_size = 25, epochs = total_epochs, validation_split = .2)
	plot_model_history(model_info)

	model_json = model.to_json()
	with open("model_rnn_lstm.json", "w") as json_file:
		json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights("model_rnn_lstm.h5")
		print("Saved model to disk")