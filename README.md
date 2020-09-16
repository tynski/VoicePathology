# Voice pathology detection
This is my bachelor thesis project. Voice pathology detection is patient condition diagnostics system based on Machine Learning algorithms. Vanilla version was based on logisitc regression and random forest, though when I discovered CNN's and computer vision I thought it would be nice to test deep learning approach. To conclude project contains to aproches to voice pathology detection problem.

Responisiblities:
* Built machine learning project pipeline: data analysis, features creation, model preparation, validation 

## Classic machine learning aproach
I used ML for detect healthy and ill patients. Actually  I used two algorithms logistic regression and random forest with overall accuracy ~70%.

## CNN
Convolutional neural network utlized to classified patients based on extracted spectograms, overall accuracy ~70%.


## Installation
Just clone the repository. Required dependencies are given in **requirements.txt** file:

`pip install -r requirements.txt`

## Classifiaction
The classification workflow is presented in Jupyter Notebooks:
* [classic machine learnig](voice_pathology/classic_ml/classification.ipynb) `voice_pathology/classic_ml/classification.ipynb`
* [CNN](voice_pathology/cnn/classification.ipynb) `voice_pathology/cnn/classification.ipynb`

## Dataset
Classification is based on [Saarbruecken Voice Database](http://www.stimmdatenbank.coli.uni-saarland.de/help_en.php4) a collection of voice recordings from more than 2000 persons. There are two Jupyter Notebooks presenting data preparation:
* [classic machine learnig](voice_pathology/classic_ml/audio_analysis.ipynb) `voice_pathology/classic_ml/audio_analysis.ipynb`
* [CNN](voice_pathology/cnn/audio_analysis.ipynb) `voice_pathology/cnn/audio_analysis.ipynb`
