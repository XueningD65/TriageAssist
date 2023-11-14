import pandas as pd

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import (
    count,
    selection_rate,
    equalized_odds_difference,
    false_positive_rate,
    false_negative_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import EqualizedOdds
from sklearn.model_selection import train_test_split
import warnings

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn import tree

warnings.simplefilter("ignore")

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

rand_seed = 1234
np.random.seed(rand_seed)

import helper_func as f

####### Load Dataset #######
df = pd.read_csv('heart.csv', header = 0)
df.columns = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope','HeartDisease']

df['Sex'] = df.Sex.map({'F': 2, 'M': 1})
Y, A = df.loc[:, "HeartDisease"], df.loc[:, "Sex"]
X = pd.get_dummies(df.drop(columns=["HeartDisease", "Sex"]), dtype = float)

A_str = A.map({1: "male", 2: "female"})
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X, Y, A_str, test_size=0.2, stratify=Y
)

X_train, y_train, A_train = f.resample_training_data(X_train, y_train, A_train)

############################################
# Use SHAP to analyze LightGBM and XGBoost #
############################################

# train an XGBoost model
# import xgboost
# import shap

# from xgboost import XGBClassifier
# xg = XGBClassifier()
# xg.fit(X_train, y_train)
# y_pred = xg.predict(X_test)

# from sklearn.metrics import confusion_matrix
# cm_test = confusion_matrix(y_pred, y_test)

# y_pred_train = xg.predict(X_train)

# for i in range(0, len(y_pred_train)):
#     if y_pred_train[i]>= 0.5:       # setting threshold to .5
#        y_pred_train[i]=1
#     else:  
#        y_pred_train[i]=0
       
# cm_train = confusion_matrix(y_pred_train, y_train)
# print("XGBoost:     Training accuracy: {:.2f}% | Test accuracy: {:.2f}% | F1 score: {:.3f}".format(accuracy_score(y_pred_train, y_train)*100, 
#                                                                                   accuracy_score(y_pred, y_test)*100, 
#                                                                                   f1_score(y_pred, y_test))) 
# print("Train Confusion matrix", cm_train.ravel(), "| Test Confusion Matrix", cm_test.ravel())

# # explain the model's predictions using SHAP
# # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
# explainer = shap.Explainer(xg)
# shap_values = explainer(X_train)

# # visualize the first prediction's explanation
# shap.plots.waterfall(shap_values[0])

# plt.show()

# # summarize the effects of all the features
# shap.plots.bar(shap_values)

# plt.show()

# # train an LightGBM model
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# import lightgbm as lgb

# d_train = lgb.Dataset(X_train, label = y_train)
# params = {}

# clf = lgb.train(params, d_train, 100)
# #Prediction
# y_pred = clf.predict(X_test)
# #convert into binary values
# for i in range(0, len(y_pred)):
#     if y_pred[i]>= 0.5:       # setting threshold to .5
#        y_pred[i]=1
#     else:  
#        y_pred[i]=0
       
# cm_test = confusion_matrix(y_pred, y_test)

# y_pred_train = clf.predict(X_train)

# for i in range(0, len(y_pred_train)):
#     if y_pred_train[i]>= 0.5:       # setting threshold to .5
#        y_pred_train[i]=1
#     else:  
#        y_pred_train[i]=0
       
# cm_train = confusion_matrix(y_pred_train, y_train)

# print("lightgbm:      Training accuracy: {:.2f}% | Test accuracy: {:.2f}% | F1 score: {:.3f}".format(accuracy_score(y_pred_train, y_train)*100, 
#                                                                                   accuracy_score(y_pred, y_test)*100, 
#                                                                                   f1_score(y_pred, y_test))) 
# print("Train Confusion matrix", cm_train.ravel(), "| Test Confusion Matrix", cm_test.ravel())

# # explain the model's predictions using SHAP
# # (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
# explainer = shap.Explainer(clf)
# shap_values = explainer(X_train)

# # visualize the first prediction's explanation
# shap.plots.waterfall(shap_values[0])

# plt.show()

# # summarize the effects of all the features
# shap.plots.bar(shap_values)

# plt.show()

############################################
#     Use SHAP to analyze other models     #
############################################

import sklearn
import shap
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# print the JS visualization code to the notebook
shap.initjs()

# train a SVM classifier
classifier = SVC(kernel = 'linear', probability=True)
classifier.fit(X_train, y_train)

train_pred = classifier.predict(X_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


cm_test = confusion_matrix(y_pred, y_test)
cm_train = confusion_matrix(train_pred, y_train)

print("SVM linear:    Training accuracy: {:.2f}% | Test accuracy: {:.2f}% | F1 score: {:.3f}".format(accuracy_score(train_pred, y_train)*100, 
                                                                                  accuracy_score(y_pred, y_test)*100, 
                                                                                  f1_score(y_pred, y_test))) 
print("Coefficients:", classifier.coef_)
print("Train Confusion matrix", cm_train.ravel(), "| Test Confusion Matrix", cm_test.ravel())

# use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(classifier.predict_proba, X_train, link="logit")
shap_values = explainer.shap_values(X_test, nsamples=100)

# plot the SHAP values for the Setosa output of the first instance
shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:], link="logit")
plt.show()