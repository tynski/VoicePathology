# Voice pathology detection
This is my bachelor thesis project. Voice pathology detection is a patient condition diagnostics system based on Machine Learning algorithms. Vanilla version was based on logistic regression and random forest, though when I discovered CNN's and computer vision I thought it would be nice to test a deep learning approach. To conclude the project contains approaches to voice pathology detection problem.

Responsibilities:
* Built machine learning project pipeline: data analysis, features creation, model preparation, validation 

## Classic machine learning approach
I used ML to detect healthy and ill patients. Actually, I used two algorithms logistic regression and random forest with overall accuracy ~70%.

## CNN
Convolutional neural network utilized to classified patients based on extracted spectrograms, overall accuracy ~70%.


## Installation
Just clone the repository. Required dependencies are given in **requirements.txt** file:

`pip install -r requirements.txt`

## Classifiaction
The classification workflow is presented in Jupyter Notebooks:
* [classic machine learning](voice_pathology/classic_ml/classification.ipynb) `voice_pathology/classic_ml/classification.ipynb`
* [CNN](voice_pathology/cnn/classification.ipynb) `voice_pathology/cnn/classification.ipynb`

## Dataset
Classification is based on [Saarbruecken Voice Database](http://www.stimmdatenbank.coli.uni-saarland.de/help_en.php4) a collection of voice recordings from more than 2000 persons. There are two Jupyter Notebooks presenting data preparation:
* [classic machine learning](voice_pathology/classic_ml/audio_analysis.ipynb) `voice_pathology/classic_ml/audio_analysis.ipynb`
* [CNN](voice_pathology/cnn/audio_analysis.ipynb) `voice_pathology/cnn/audio_analysis.ipynb`
