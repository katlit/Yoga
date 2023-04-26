import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sn


from sklearn import metrics
from sklearn.model_selection import train_test_split, ParameterGrid, GridSearchCV
from sklearn.linear_model import LogisticRegression as LR


#Classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


#path = '../data/'
PerfectTrainLandmarks_df = pd.read_csv("PerfectTrainLandmarks_df.csv")
PerfectTestLandmarks_df = pd.read_csv("PerfectTestLandmarks_df.csv")

ColList = PerfectTrainLandmarks_df.columns
sub = 'vis'
ColListVis = [s for s in ColList if sub in s]

X_Train_df = PerfectTrainLandmarks_df[ColListVis]
Y_Train_df = PerfectTrainLandmarks_df['label of class_82']
X_Train = X_Train_df.to_numpy()
Y_Train = Y_Train_df.to_numpy()

X_Test_df = PerfectTestLandmarks_df[ColListVis]
Y_Test_df = PerfectTestLandmarks_df['label of class_82']
X_Test = X_Test_df.to_numpy()
Y_Test = Y_Test_df.to_numpy()

parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
              {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
              {'kernel': ['poly'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
              {'kernel': ['sigmoid'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]

optimized_svm = GridSearchCV(SVC(), parameters)
svm_model = optimized_svm

optimized_svm.estimator

SVC()

svm_model.fit(X_Train, Y_Train)

path = ''

predictions  = svm_model.predict(X_test)
np.savetxt(path + 'SVM_predictions.txt', predictions, fmt='%d')

score = metrics.accuracy_score(X_test, predictions)
np.savetxt(path + 'SVM_score.txt', score, fmt='%d')

cm = metrics.confusion_matrix(predictions, X_test)
cm_df = pd.DataFrame(cm)
cm_df.to_csv(path + "SVM_Confusion.csv")
