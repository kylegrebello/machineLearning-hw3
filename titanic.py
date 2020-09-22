import pandas as pd
import random as rnd
import numpy as np
import math
import operator
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

#Read in data
trainingData = pd.read_csv('train.csv')
testingData = pd.read_csv('test.csv')
#Combine data for processing that needs to happen on both sets
combinedData = [trainingData, testingData]

#Preprocess Data

##Step 1 - From HW1 we know that features Cabin and Ticket aren't useful, so we will drop them immediately
#We will also drop PassengerID from both as well since it is not an important feature
trainingData = trainingData.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
testingData = testingData.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
#Update the combined data array
combinedData = [trainingData, testingData]

##Step 2 - Fill missing Data
#The Embarked feature in the trainingData set is missing two values, so fill with the most common
mostCommonEmbarkationPort = trainingData.Embarked.dropna().mode()[0]
trainingData['Embarked'] = trainingData['Embarked'].fillna(mostCommonEmbarkationPort)

#The 'Fare' feature in the testingData set is missing one value, so fill with the median of the feature
medianFare = testingData['Fare'].dropna().median()
testingData['Fare'] = testingData['Fare'].fillna(medianFare)

##Step 3 - Loop through the data and preprocess
for data in combinedData:
	#Create new feature called NamePrefix and extract the title from each name (ie Mr., Mrs., ...)
	data['NamePrefix'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
	#Combine prefix's with same meaning but different spelling (mlle & miss, ms & miss, mme & mrs)
	data['NamePrefix'] = data['NamePrefix'].replace('Mlle', 'Miss')
	data['NamePrefix'] = data['NamePrefix'].replace('Ms', 'Miss')
	data['NamePrefix'] = data['NamePrefix'].replace('Mme', 'Mrs')
	#Replace the names of all entries with less than 7 people with the value 'Other'
	data['NamePrefix'] = data['NamePrefix'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir'], 'Other')
	#Convert 'NamePrefix' categorical feature to ordinal
	data['NamePrefix'] = data['NamePrefix'].map( {'Other' : 0, 'Master' : 1, 'Miss' : 2, 'Mr' : 3, 'Mrs' : 4} )

	#Convert 'Sex' categorical feature to ordinal
	data['Sex'] = data['Sex'].map( {'male' : 0, 'female' : 1} ).astype(int)
	#Rename to 'Gender'
	data.rename(columns={'Sex': 'Gender'}, inplace=True)

	#Need to fill in missing Age values
	#From HW1 we found a correlation between Age, Gender, and Pclass by plotting six graphs
	#We are going to fill in missing values by using the median value of the Age based on the position in the Pclass/Gender chart we created
	for i in range(0, 2):
		for j in range(0, 3):
			ageFromCorrelatedChart = data[(data['Gender'] == i) & (data['Pclass'] == j+1)]['Age'].dropna()
			data.loc[(data.Age.isnull()) & (data.Gender == i) & (data.Pclass == j+1), 'Age'] = int( (ageFromCorrelatedChart.median() * 2.0) + 0.5) / 2.0
	data['Age'] = data['Age'].astype(int)

	#Convert 'Age' values into ordinal data based on Age Groups found in HW1
	data.loc[ (data['Age'] <= 16), 'Age'] = 0
	data.loc[ (data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
	data.loc[ (data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
	data.loc[ (data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
	data.loc[ (data['Age'] > 64), 'Age'] = 4

	#Create new feature 'Alone' to indicate if a person is travelling alone or not.
	data['Alone'] = 0
	#Set the feature to 1 (Meaning traveller is alone) if 'SibSp' + 'Parch' == 0
	data.loc[(data['SibSp'] + data['Parch']) == 0, 'Alone'] = 1

	#Convert 'Embarked' categorical feature to ordinal
	data['Embarked'] = data['Embarked'].map( {'C' : 0, 'Q' : 1, 'S' : 2} ).astype(int)

	#Convert 'Fare' values into ordinal data
	#Fare Groups found by doing a qcut on the data and plotting it against the 'Survival' feature
	#The four groups was selected because it gave a good data split and a group with 58% survival rate
	data.loc[ (data['Fare'] <= 7.91), 'Fare'] = 0
	data.loc[ (data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
	data.loc[ (data['Fare'] > 14.454) & (data['Fare'] <= 31.0), 'Fare'] = 2
	data.loc[ (data['Fare'] > 31.0), 'Fare'] = 3
	data['Fare'] = data['Fare'].astype(int)

##Step 4 - Drop unimportant features
#Dropping Name, SibSp, Parch features
trainingData = trainingData.drop(['Name', 'SibSp', 'Parch'], axis=1)
testingData = testingData.drop(['Name', 'SibSp', 'Parch'], axis=1)
#Update the combined data array
combinedData = [trainingData, testingData]

##Step 5 - Train and classify data
print("Preprocessed Training Data")
print(trainingData.head(5))

print("Preprocessed Testing Data")
print(testingData.head(5))

#Split data for classification
X = trainingData.drop(['Survived'], axis=1)
Y = trainingData['Survived']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

class KNN():
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_test, k):
		predictions = []
		for row in X_test.itertuples():
			label = self.closestLabel(row, k)
			predictions.append(label)
		return predictions

	def closestLabel(self, row, k):
		distances = []
		for trainRow in self.X_train.itertuples():
			distances.append((trainRow[0], self.distanceBetweenRows(row, trainRow)))
		distances = sorted(distances, key=lambda x:x[1])[0:k]
		kIndeces = []
		for i in range(k):
			kIndeces.append(distances[i][0])
		kLabels = []
		for i in kIndeces:
			kLabels.append(self.y_train[i])
		counter = Counter(kLabels)
		return counter.most_common()[0][0]

	def distanceBetweenRows(self, row1, row2):
		distance = 0
		for i in range(len(row1) - 1):
			distance += pow((row1[i + 1] - row2[i + 1]), 2)
		return math.sqrt(distance)

#KNN
kVals = [3, 5, 7, 9, 11, 13, 15, 17, 19, 25, 50, 100]
knn = KNN()
knn.fit(x_train, y_train)
knnAccuracy = []
x = 1
kValsLen = len(kVals)
for k in kVals:
	predictions = knn.predict(x_test, k)
	knnAccuracy.append(accuracy_score(y_test, predictions))
	print("Percent done: {}%".format(round((100.0 / kValsLen) * x, 2)))
	x += 1

plt.plot(kVals, knnAccuracy)
plt.xlabel('Value of K')
plt.ylabel('Accuracy')
plt.show()

print(knnAccuracy)

#Naive Bayes
nb = GaussianNB()
nb.fit(x_train, y_train)
Y_pred = nb.predict(x_test)
print()
print()
print("Naive Bayes classifier")
score = cross_val_score(nb, x_train, y_train, cv=5, scoring='accuracy')
precision = cross_val_score(nb, x_train, y_train, cv=5, scoring='precision')
recall = cross_val_score(nb, x_train, y_train, cv=5, scoring='recall')
f1Score = cross_val_score(nb, x_train, y_train, cv=5, scoring='f1')

print("Accuracy: {}".format(round(score.mean() * 100, 2)))
print("Precision: {}".format(round(precision.mean() * 100, 2)))
print("Recall: {}".format(round(recall.mean() * 100, 2)))
print("F1 Score: {}".format(round(f1Score.mean() * 100, 2)))
