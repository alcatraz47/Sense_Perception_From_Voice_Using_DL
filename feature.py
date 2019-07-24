import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from librosa import display
from keras.utils import to_categorical
from librosa.feature import melspectrogram
from sklearn.preprocessing import LabelEncoder

class Feature:
	def __init__(self, working_directory, training_directory, happy_path, sad_path, confused_path, neutral_path):
		
		#directory
		self.working_directory = working_directory
		self.training_directory = training_directory
		self.happy_path = happy_path
		self.sad_path = sad_path
		self.confused_path = confused_path
		self.neutral_path = neutral_path

	#feature extractor
	#now feature extraction and label function for whole dataset
	def feature_extraction(self, categories, sampling_rate, resampling_type):
		emotion_feature_label = []
		for category in categories:
			class_path = os.path.join(self.training_directory, category)
			for wav_files in os.listdir(class_path):
				data, sampling_rate = librosa.load(os.path.join(class_path, wav_files), mono = True, sr = sampling_rate, res_type = resampling_type)
				mel_power_spectrogram_feature = melspectrogram(data, sr = sampling_rate)
				label = categories.index(category)
				emotion_feature_label.append([mel_power_spectrogram_feature, label])
		return emotion_feature_label

	def feature_label_separator(self, emotion_feature_label, sampling_rate):
		X_train , y_train = [], []
		for feature, label in emotion_feature_label:
			X_train.append(feature)
			y_train.append(label)

		#encoding
		encoder = LabelEncoder()
		encoded_y = encoder.fit_transform(y_train)
		one_hot_y_train = to_categorical(encoded_y)

		return X_train, one_hot_y_train

if __name__ == '__main__':

	working_directory = os.getcwd()
	training_directory = os.path.join(working_directory, 'train')
	happy_path = os.path.join(training_directory, 'happy')
	sad_path = os.path.join(training_directory, 'sad')
	confused_path = os.path.join(training_directory, 'confused')
	neutral_path = os.path.join(training_directory, 'neutral')

	feature = Feature(working_directory, training_directory, happy_path, sad_path, confused_path, neutral_path)

	CLASSES = ['happy', 'sad']
	emotion_feature_label = feature.feature_extraction(CLASSES, 16000, 'kaiser_best')
	X_train, y_train = feature.feature_label_separator(emotion_feature_label, 16000)
	print('X_train[:5]:')
	print(X_train[:5])
	print('y_train[:5]:')
	print(y_train[:5])