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

def run_classifier(train, test, train_labels, test_labels, classifier_class):
    startt = time.time()
    classifier_class.fit( train, train_labels )

    probas = classifier_class.predict_proba(test)    
    results = classifier_class.predict( test)
    log_loss = metrics.log_loss(test_labels, probas)
    score = log_loss
    #score = classifier_class.score( test, test_labels )
    confmat = metrics.confusion_matrix(test_labels, results)
    confmat = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis] #normalize
    duration = time.time() - startt
    return (type(classifier_class).__name__, score, duration, confmat)


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    LR(multi_class='ovr')]

classifier_rslt_all = [run_classifier(Xtrain, Xtest, train_labels, test_labels, classifier) for classifier in classifiers]

classifier_names = [e[0] for e in classifier_rslt_all]
confmats = [e[3] for e in classifier_rslt_all]


results = pd.DataFrame.from_records([ (e[0],e[1],e[2]) for e in classifier_rslt_all], columns=['Classifier', 'Score','Duration'])    


path = ''

results.to_csv(path + "Mixed_Classification_Results.csv")
