"""
Alghoritm iterates for every audio file in given directory.
Each audio file is processed in order to extract parameters.
Finally all paremters vectors for each signal are saved to CSV file.
"""
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
    fulldir = os.path.join(os.getcwd(), datadir)
    df2Csv = input('Enter file name to save CSV:') + '.csv'
    allfiles = os.listdir(fulldir)
    columns = ['MFCC1',
               'MFCC2',
               'MFCC3',
               'MCCC4',
               'MFCC5',
               'MFCC6',
               'MFCC7',
               'MFCC8',
               'MFCC9',
               'MFCC10',
               'RMS',
               'MAX',
               'MIN']
    cases = pd.DataFrame(columns=columns)
    for filename in allfiles:
        filepath = os.path.join(datadir, filename)
        cases.loc[len(cases)] = parameters(filepath)
    cases['Status'] = patientstatus
    cases.to_csv(df2Csv, index=False)


def parameters(filepath):
    y, sr = librosa.load(filepath)
    if(~(np.isnan(np.sum(y)))):
        hop_length = 512
        n_mfcc = 10
        mfcc = librosa.feature.mfcc(y=y,
                                    sr=sr,
                                    hop_length=hop_length,
                                    n_mfcc=n_mfcc)
        parameters = []
        for i in range(n_mfcc):
            parameters.append(np.mean(mfcc[i]))
        rms = librosa.feature.rmse(y)
        RMS = np.mean(rms)
        parameters.append(RMS)
        MAX = np.amax(y)
        parameters.append(MAX)
        MIN = np.amin(y)
        parameters.append(MIN)
        return parameters
