import pandas as pd
import random as rnd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

#Read in data
trainingData = pd.read_csv('train.csv')
testingData = pd.read_csv('test.csv')

trainingData = trainingData.drop(['ID', 'Date', 'Opponent'], axis=1)
testingData = testingData.drop(['ID', 'Date', 'Opponent'], axis=1)
#Combine data for processing that needs to happen on both sets
combinedData = [trainingData, testingData]

#Preprocess Data

for dataset in combinedData:
	dataset['Is_Home_or_Away'] = dataset['Is_Home_or_Away'].map( {'Home' : 1, 'Away' : 2} ).astype(int)
	dataset['Is_Opponent_in_AP25_Preseason'] = dataset['Is_Opponent_in_AP25_Preseason'].map( {'In' : 1, 'Out' : 2} ).astype(int)
	dataset['Media'] = dataset['Media'].map( {'1-NBC' : 1, '2-ESPN' : 2, '3-FOX' : 3, '4-ABC' : 4, '5-CBS' : 5} ).astype(int)
	dataset['Label'] = dataset['Label'].map( {'Lose' : 0, 'Win' : 1} ).astype(int)

	dataset.info()

testingLabels = testingData['Label']
testingData = testingData.drop(['Label'], axis=1)
combinedData = [trainingData, testingData]

print(trainingData[['Is_Home_or_Away', 'Label']].groupby(['Is_Home_or_Away'], as_index=False).mean().sort_values(by='Label', ascending=False))
print(trainingData[['Is_Opponent_in_AP25_Preseason', 'Label']].groupby(['Is_Opponent_in_AP25_Preseason'], as_index=False).mean().sort_values(by='Label', ascending=False))
print(trainingData[['Media', 'Label']].groupby(['Media'], as_index=False).mean().sort_values(by='Label', ascending=False))

#Split data for classification
X_train = trainingData.drop(['Label'], axis=1)
y_train = trainingData['Label']
X_test = testingData.copy()

#Origional Labels
print()
print()
origionalLabels = []
for y in testingLabels:
	if (y == 1):
		origionalLabels.append('Win')
	else:
		origionalLabels.append('Lose')
print(origionalLabels)

#Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
Y_pred = nb.predict(X_test)
print()
print()
print("Naive Bayes classifier")
print("Accuracy: {}".format(round(accuracy_score(testingLabels, Y_pred) * 100, 2)))
print("Precision: {}".format(round(precision_score(testingLabels, Y_pred, average='macro') * 100, 2)))
print("Recall: {}".format(round(recall_score(testingLabels, Y_pred, average='macro') * 100, 2)))
print("F1 Score: {}".format(round(f1_score(testingLabels, Y_pred, average='macro') * 100, 2)))
#print(Y_pred)
predictedLabels = []
for y in Y_pred:
	if (y == 1):
		predictedLabels.append('Win')
	else:
		predictedLabels.append('Lose')

print()
print()
print(predictedLabels)

#KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
print()
print()
print("KNN classifier")
print("Accuracy: {}".format(round(accuracy_score(testingLabels, Y_pred) * 100, 2)))
print("Precision: {}".format(round(precision_score(testingLabels, Y_pred, average='macro') * 100, 2)))
print("Recall: {}".format(round(recall_score(testingLabels, Y_pred, average='macro') * 100, 2)))
print("F1 Score: {}".format(round(f1_score(testingLabels, Y_pred, average='macro') * 100, 2)))
#print(Y_pred)
predictedLabels = []
for y in Y_pred:
	if (y == 1):
		predictedLabels.append('Win')
	else:
		predictedLabels.append('Lose')

print()
print()
print(predictedLabels)
