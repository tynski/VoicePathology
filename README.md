# Voice pathology detection
This is my bachelor thesis project. Voice pathology detection is patient condition diagnostics system based on Machine Learning algorithms. Vanilla version was based on logisitc regression and random forest, though when I discovered CNN's and computer vision I thought it would be nice to test deep learning approach. To conclude project contains to aproches to voice pathology detection problem.

Responisiblities:
* Built machine learning project pipeline: data analysis, features creation, model preparation, validation 

## Classic machine learning aproach
I used ML for detect healthy and ill patients. Actually  I used two algorithms logistic regression and random forest with overall accuracy ~70%. I think the problem is signal processing due to get features, so I try different approach and test CNN in action.

accuracy = 0.7163636

## CNN

## Installation
Just clone the repository. Required dependencies are given in **requirements.txt** file:

`pip install -r requirements.txt`

## Usage
The whole workflow is presented in Jupyter Notebooks:
* [classic machine learnig](voice_pathology/classicML/classification.ipynb) `voice_pathology/classicML/classification.ipynb`
* [CNN](voice_pathology/CNN/Classification.ipynb) `voice_pathology/CNN/Classification.ipynb`

## TODO
* documentation / README
* restructurize project architecture
* run check if jupyter notebooks are working
