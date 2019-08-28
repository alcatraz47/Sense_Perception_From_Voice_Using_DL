import os
import numpy as np
import pickle
import librosa
import matplotlib.pyplot as plt
from librosa import display
from keras.utils import to_categorical
from librosa.feature import mfcc
from librosa.feature import melspectrogram
from sklearn.preprocessing import LabelEncoder

class Feature:

	#feature extractor

	def feature_extraction_mfcc(self, training_directory, categories, sampling_rate, resampling_type):
		emotion_feature_label = []
		for category in categories:
			class_path = os.path.join(training_directory, category)
			for wav_files in os.listdir(class_path):
				data, sampling_rate = librosa.load(os.path.join(class_path, wav_files), sr = sampling_rate, res_type = resampling_type)
				mfcc_feature = np.mean(mfcc(y = data, sr = sampling_rate, n_mfcc = 32).T, axis = 0)
				label = categories.index(category)
				mfcc_feature = np.asarray(mfcc_feature)
				emotion_feature_label.append([mfcc_feature, label])
		return emotion_feature_label

	#now feature extraction and label function for whole dataset
	def feature_extraction(self, training_directory, categories, sampling_rate, resampling_type):
		emotion_feature_label = []
		for category in categories:
			class_path = os.path.join(training_directory, category)
			for wav_files in os.listdir(class_path):
				data, sampling_rate = librosa.load(os.path.join(class_path, wav_files), mono = True, sr = sampling_rate, res_type = resampling_type)
				mel_power_spectrogram_feature = np.mean(melspectrogram(data, sr = sampling_rate, n_mels = 32).T, axis = 0)
				label = categories.index(category)
				mel_power_spectrogram_feature = np.asarray(mel_power_spectrogram_feature)
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
	CLASSES = ['Old_Talker_Angry', 'Old_Talker_Disgust', 'Old_Talker_Fear', 'Old_Talker_Happy', 'Old_Talker_Neutral', 'Old_Talker_Pleasent_Surprise', 'Old_Talker_Sad']
	training_directory = os.path.join(working_directory, 'train')
	training_directory = os.path.join(training_directory, 'Toronto_Dataset')
	old_Talker_Angry = os.path.join(training_directory, 'Old_Talker_Angry')
	old_Talker_Disgust = os.path.join(training_directory, 'Old_Talker_Disgust')
	old_Talker_Fear = os.path.join(training_directory, 'Old_Talker_Fear')
	old_Talker_Happy = os.path.join(training_directory, 'Old_Talker_Happy')
	old_Talker_Neutral = os.path.join(training_directory, 'Old_Talker_Neutral')
	old_Talker_Pleasent_Surprise = os.path.join(training_directory, 'Old_Talker_Pleasent_Surprise')
	old_Talker_Sad = os.path.join(training_directory, 'Old_Talker_Sad')

	feature = Feature()
	emotion_feature_label = feature.feature_extraction_mfcc(training_directory, CLASSES, 42000, 'kaiser_best')

	X_train, y_train = feature.feature_label_separator(emotion_feature_label, 42000)

	out_file = open("X_train_mfcc_42000.pickle", "wb")
	pickle.dump(X_train, out_file)
	out_file.close()

	out_file = open("y_train_mfcc_42000.pickle", "wb")
	pickle.dump(y_train, out_file)
	out_file.close()

	print('X_train[:5]:')
	print(X_train[:5])
	print('y_train[:5]:')
	print(y_train[:5])