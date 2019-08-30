import os
import numpy as np
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
import librosa
import matplotlib.pyplot as plt
from librosa import display
from keras.utils import to_categorical
from librosa.feature import mfcc
from librosa.feature import melspectrogram
from sklearn.preprocessing import LabelEncoder


def load_test_data(training_directory, categories, sampling_rate, resampling_type):
		emotion_feature_label = []
		for category in categories:
			class_path = os.path.join(training_directory, category)
			for wav_files in os.listdir(class_path):
				data, sampling_rate = librosa.load(os.path.join(class_path, wav_files), mono = True, sr = sampling_rate, res_type = resampling_type)
				# mel_power_spectrogram_feature = np.mean(melspectrogram(data, sr = sampling_rate, n_mels = 128, n_fft = 2048, hop_length = 512).T, axis = 0)
				label = categories.index(category)
				# mel_power_spectrogram_feature = np.asarray(mel_power_spectrogram_feature)
				data = np.asarray(data)
				emotion_feature_label.append([data, label])
		return emotion_feature_label

def feature_label_separator(self, emotion_feature_label, sampling_rate):
		X_train , y_train = [], []
		for feature, label in emotion_feature_label:
			X_test.append(feature)
			y_test.append(label)

		#encoding
		encoder = LabelEncoder()
		encoded_y = encoder.fit_transform(y_test)
		one_hot_y_test = to_categorical(encoded_y)

		return X_test, one_hot_y_test

if __name__ == '__main__':

	CLASSES = ['Younger_Talker_Angry', 'Younger_Talker_Disgust', 'Younger_Talker_Fear', 'Younger_Talker_Happy', 'Younger_Talker_Neutral', 
	'Younger_Talker_Pleasent_Surprise', 'Younger_Talker_Sad']
	working_directory = os.getcwd()
	training_directory = os.path.join(working_directory, 'train/Toronto_Dataset')
	young_Talker_Angry = os.path.join(training_directory, 'Younger_Talker_Angry')
	young_Talker_Fear = os.path.join(training_directory, 'Younger_Talker_Fear')
	young_Talker_Neutral = os.path.join(training_directory, 'Younger_Talker_Neutral')
	young_Talker_Pleasant_Surprise = os.path.join(training_directory, 'Younger_Talker_Pleasant_Surprise')
	young_Talker_Sad = os.path.join(training_directory, 'Younger_Talker_Sad')
	young_Talker_Disgust = os.path.join(training_directory, 'Younger_Walker_Disgust')
	young_Talker_Happy = os.path.join(training_directory, 'Younger_Talker_Happy')

