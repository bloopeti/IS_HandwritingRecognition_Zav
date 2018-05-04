"""
================================
Recognizing hand-written letters
================================

An example showing how scikit-learn can be used to recognize images of
hand-written letters.

"""
print(__doc__)

# Author: Peter-Tibor Zavaczki
# Inspiration: Gael Varoquaux <gael dot varoquaux at normalesup dot org>, "Recognizing hand-written digits", http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import numpy / pandas for csv loading
import numpy as np
# import pandas as pd

# Import classifiers and performance metrics
from sklearn import svm, metrics

# The letters dataset
lettersTrainRaw = np.loadtxt(fname = "./emnist-letters-train.csv", delimiter = ',')
lettersTrainTarget = lettersTrainRaw[:, 0]
lettersTrainData = lettersTrainRaw[:, 1:]

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size.
# For these images, we know which digit they represent: it is given in the
# 'target' of the dataset.
images_and_labels = list(zip(lettersTrainData, lettersTrainTarget))
#for index, (image, label) in enumerate(images_and_labels[:4]):
#    plt.subplot(2, 4, index + 1)
#    plt.axis('off')
#    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image,
# to turn the data in a (samples, feature) matrix:
n_samplesTrain = len(lettersTrainData)
dataTrain = lettersTrainData.reshape((n_samplesTrain, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

print("1")

# We learn the digits on the first half of the digits
classifier.fit(dataTrain[:n_samplesTrain // 1], lettersTrainTarget[:n_samplesTrain // 1])

print("2")

# Now predict the value of the digit on the second half:
lettersTestRaw = np.loadtxt(fname = "./emnist-letters-train.csv", delimiter = ',')
lettersTestTarget = lettersTestRaw[:, 0]
lettersTestData = lettersTestRaw[:, 1:]

n_samplesTest = len(lettersTestData)
dataTest = lettersTestData.reshape((n_samplesTest, -1))
predicted = classifier.predict(dataTest)
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(lettersTestTarget, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(lettersTestTarget, predicted))

images_and_predictions = list(zip(lettersTestTarget, predicted))
#for index, (image, prediction) in enumerate(images_and_predictions[:4]):
#    plt.subplot(2, 4, index + 5)
#    plt.axis('off')
#    plt.imshow(image, cmap=plt.cm.gexiray_r, interpolation='nearest')
#    plt.title('Prediction: %i' % prediction)

plt.show()

