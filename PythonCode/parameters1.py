"""
Alghoritm iterates for every audio file in given directory.
Each audio file is processed in order to extract parameters.
Finally all paremters vectors for each signal are saved to CSV file.
"""

import os
from scipy.io import wavfile
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.stats import skew


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
    columns = ['MAX',
               'MIN',
               'RMS',
               'KURTOSIS',
               'SKEWNESS']
    cases = pd.DataFrame(columns=columns)
    for filename in allfiles:
        filepath = os.path.join(datadir, filename)
        cases.loc[len(cases)] = returnparametersdf(filepath)
    cases['Status'] = patientstatus
    cases.to_csv(df2Csv, index=False)


def returnparametersdf(filepath):
    signal = wavfile.read(filepath)[1]
    if(~(np.isnan(np.sum(signal)))):
        signal = signal.tolist()
        MAX = np.amax(signal)
        MIN = np.amin(signal)
        RMS = np.sqrt(np.sum(np.power(signal, 2)) / len(signal))
        KURTOSIS = kurtosis(signal)
        SKEWNESS = skew(signal)
        return np.array([MAX, MIN, RMS, KURTOSIS, SKEWNESS])
