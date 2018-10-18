import numpy as np

def voiceParameters(signal):
	N = len(signal)
	Parameters = [RMS(signal), mean(signal), energy(signal), power(signal, N), min(signal), max(signal)]
	return Parameters

def RMS(signal):
	return np.sqrt(np.mean(np.power(signal, 2)))

def mean(signal):
	return np.mean(signal)

def energy(signal):
	return np.sum(np.power(signal, 2))

def power(signal, N):
	return np.sum(np.power(signal, 2)/(2*N + 1))