from __future__ import print_function
import os
import numpy as np
import pandas as pd
import librosa


def parameterstocsv(datadir, patientstatus):
    """
    Parameters
    ----------
    datadir
        Audio files directory
    patientstatus
        Healthy or pathology
    """
    df2Csv = input('Enter file name to save CSV:') + '.csv'
    allfiles = os.listdir(datadir)
    columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
    cases = pd.DataFrame(columns=columns)
    for filename in allfiles:
        filepath = os.path.join(datadir, filename)
        cases.loc[len(cases)] = parameters(filepath)
    cases['Status'] = patientstatus
    cases.to_csv(df2Csv, index=False)


def parameters(filepath):
    y, sr = librosa.load(filepath)
    hop_length = 512
    n_mfcc = 13
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    parameters = []
    for i in range(n_mfcc):
        parameters.append(np.mean(mfcc[i]))
    rms = librosa.feature.rmse(y)
    RMS = np.mean(rms)
    parameters.append(RMS)
    return parameters

"""
'/home/barti/Documents/voicePathology/voiceDataset/FemaleHealthy'
"""