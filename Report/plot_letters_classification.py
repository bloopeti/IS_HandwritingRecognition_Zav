"""
================================
Recognizing hand-written letters
================================

An example showing how scikit-learn can be used to recognize images 
of hand-written letters.

"""
print(__doc__)

# Author: Peter-Tibor Zavaczki
# Inspiration: Gael Varoquaux <gael dot varoquaux at normalesup dot org>, "Recognizing hand-written digits", http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py

# Import numpy / pandas for csv loading
import numpy as np

# Import classifiers and performance metrics
from sklearn import svm, metrics

# Import pandas or joblib for storing data
# import pandas as pd
from sklearn.externals import joblib

# The letters dataset
lettersTrainRaw = np.loadtxt(fname = "./emnist-letters-train.csv", delimiter = ',')
lettersTrainTarget = lettersTrainRaw[:, 0]
lettersTrainData = lettersTrainRaw[:, 1:]

# To apply a classifier on this data, we need to flatten the image,
# to turn the data in a (samples, feature) matrix:
n_samplesTrain = len(lettersTrainData)
dataTrain = lettersTrainData.reshape((n_samplesTrain, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

print("1")

# We learn the letters in the training dataset
classifier.fit(dataTrain[:n_samplesTrain // 1], lettersTrainTarget[:n_samplesTrain // 1])

print("2")

# Now predict the value of the letters in the tresting set
lettersTestRaw = np.loadtxt(fname = "./emnist-letters-train.csv", delimiter = ',')
lettersTestTarget = lettersTestRaw[:, 0]
lettersTestData = lettersTestRaw[:, 1:]

n_samplesTest = len(lettersTestData)
dataTest = lettersTestData.reshape((n_samplesTest, -1))
predicted = classifier.predict(dataTest)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(lettersTestTarget, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(lettersTestTarget, predicted))
