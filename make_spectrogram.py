import os
import sys

#init values
overlap = 1024
frame_length = 2048

from scipy.io import wavfile

def readAudio(audio):
    fs, amp = wavfile.read(audio)
    dt = 1/fs
    n = len(amp)
    t = dt*n

    if t > 1.0:
        amp = amp[int((t/2 - 0.5)/dt):int((t/2 + 0.5)/dt)]
        n = len(amp)
        t = dt*n
    
    return(amp, fs)

import librosa
import librosa.display
import numpy as np

def makeSpectogram(amp, fs):
    S = librosa.feature.melspectrogram(y=amp*1.0, sr=fs, n_fft=frame_length, hop_length=overlap, power=2.0)
    sepectogram = librosa.power_to_db(S,ref=np.max)
    return sepectogram.round(3).tolist()

datadir = "/home/sywi/Documents/CI/voice_pathology_detection/dataset/test"
fulldir = os.path.join(os.getcwd(), datadir)
allfiles = os.listdir(fulldir)

output = {}
output['status'] = 0

import json

for filename in allfiles:
    filepath = os.path.join(datadir, filename)
    name = filename.split(".")[0]
    amp, fs = readAudio(filepath)       
    
    output['spectogram'] = makeSpectogram(amp, fs) 

    with open(name + '.json', 'w') as outfile:
        json.dump(output, outfile)

