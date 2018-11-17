import os 
from scipy.io import wavfile
import numpy as np
import pandas as pd

def parameterstocsv(datadirectory, patientstatus):
	df2Csv = input('Enter file name:')
	df2Csv += '.csv'
	filepath = os.path.join('/home/barti/Documents/voicePathology/voiceDataset/', datadirectory)
	files = os.listdir(filepath)
	columns = ['STD','MEAN','MAX','MIN','RMS','ENERGY','POWER']
	cases = pd.DataFrame(columns = columns)
	for filename in files:
		data = returnparameters(filepath, filename)
		data = np.reshape(data, (1,len(columns)))
		df = pd.DataFrame(data = data, columns = columns)
		cases = cases.append(df)
	cases['Status'] = patientstatus
	cases.to_csv(df2Csv, index = False)

def returnparameters(filepath, filename):
	audio = wavfile.read(os.path.join(filepath, filename))[1]
	audio = audio[~np.isnan(audio)]
	return voiceparameters(audio.tolist())

def voiceparameters(signal):
	N = len(signal)
	STD = np.std(signal)
	MEAN = np.mean(signal)
	MAX = np.amax(signal)
	MIN = np.amin(signal)
	RMS = np.sqrt(np.mean(np.square(signal)))
	ENERGY = np.sum(np.power(signal, 2))
	POWER = np.sum(np.power(signal, 2)/(2*N + 1))
	return [STD,MEAN,MAX,MIN,RMS,ENERGY,POWER]
