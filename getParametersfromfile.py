import os 
from scipy.io import wavfile
import numpy as np

def ParametersFromFile(filePath):
	Files = os.listdir(filePath)
	cases = pd.DataFrame(columns= ['RMS', 'mean', 'energy', 'power', 'min', 'max'])
	for fileName in Files:
		print(returnParameters(filePath, fileName))

def returnParameters(filePath, fileName):
	audio = wavfile.read(os.path.join(filePath, fileName))[1]
	audio = audio[~np.isnan(audio)]
	return voiceParameters(audio.tolist())	

def voiceParameters(signal):
	N = len(signal)
	return [RMS(signal), mean(signal), energy(signal), power(signal, N), min(signal), max(signal)]

def RMS(signal):
	return np.sqrt(np.mean(np.power(signal, 2)))

def mean(signal):
	return np.mean(signal)

def energy(signal):
	return np.sum(np.power(signal, 2))

def power(signal, N):
	return np.sum(np.power(signal, 2)/(2*N + 1))
