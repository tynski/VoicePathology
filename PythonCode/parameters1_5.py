"""
Alghoritm iterates for every audio file in given directory.
Each audio file is processed in order to extract parameters.
Finally all paremters vectors for each signal are saved to CSV file.
"""

import os
from scipy.io import wavfile
import numpy as np
import pandas as pd


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
    columns = ['MAX',
               'MIN',
               'RMS']
    cases = pd.DataFrame(columns=columns)
    for filename in allfiles:
        filepath = os.path.join(datadir, filename)
        cases.loc[len(cases)] = returnparametersdf(filepath)
    cases['Status'] = patientstatus
    cases.to_csv(df2Csv, index=False)


def returnparametersdf(filepath):
    signal = wavfile.read(filepath)[1]
    N = len(signal)
    if(~(np.isnan(np.sum(signal)))):
        signal = signal.tolist()
        framelength = round(N / 25)
        MAX = []
        MIN = []
        RMS = []
        for i in range(framelength, N, framelength):
            start = i - framelength
            end = i
            frame = signal[start:end]
            MAX.append(np.amax(frame))
            MIN.append(np.amin(frame))
            RMS.append(np.sqrt(np.sum(np.power(frame, 2)) / framelength))
        return np.array([np.mean(MAX), np.mean(MIN), np.mean(RMS)])
