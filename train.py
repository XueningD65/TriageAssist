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

####### Global variable #######
acc = []
train_acc = []
F1s = []
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

####### Decision Tree #######
feature_names = list(X.columns)
fig, ax = plt.subplots(1,1,figsize=(20,12))
model = DecisionTreeClassifier(min_samples_split = 50, max_depth=3, random_state=21).fit(X_train, y_train)
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
Y_t = np.array(y_test.tolist())
print("Decision Tree: Training accuracy: {:.2f}% | Test accuracy: {:.2f}% | F1 score: {:.3f}".format(accuracy_score(train_pred, y_train)*100, \
                                                                                  accuracy_score(test_pred, y_test)*100,
                                                                                  f1_score(test_pred, Y_t)))

cm_test = confusion_matrix(test_pred, y_test)
cm_train = confusion_matrix(train_pred, y_train)
print("Train Confusion matrix", cm_train.ravel(), "| Test Confusion Matrix", cm_test.ravel())
acc.append(accuracy_score(test_pred, y_test)*100)
train_acc.append(accuracy_score(train_pred, y_train)*100)
F1s.append(f1_score(test_pred, Y_t)*100)

# uncomment if you want to plot the tree 
tree.plot_tree(model, feature_names = feature_names, ax=ax, fontsize=6, class_names=['No Disease', 'Has Disease'], filled=True)
plt.show()

####### Random Forest #######
feature_names = list(X.columns)
random_forest_model = RandomForestClassifier(max_depth = 8, 
                                            min_samples_split = 100,
                                            n_estimators = 10,
                                            random_state = 21).fit(X_train, y_train)

train_pred = random_forest_model.predict(X_train)
test_pred = random_forest_model.predict(X_test)
print("Random Forest: Training accuracy: {:.2f}% | Test accuracy: {:.2f}% | F1 score: {:.3f}".format(accuracy_score(train_pred, y_train)*100, 
                                                                                  accuracy_score(test_pred, y_test)*100, 
                                                                                  f1_score(test_pred, y_test))) 

cm_test = confusion_matrix(test_pred, y_test)
cm_train = confusion_matrix(train_pred, y_train)
print("Train Confusion matrix", cm_train.ravel(), "| Test Confusion Matrix", cm_test.ravel())
acc.append(accuracy_score(test_pred, y_test)*100)
train_acc.append(accuracy_score(train_pred, y_train)*100)
F1s.append(f1_score(test_pred, y_test)*100)

# for i in range(8):
#     fig, ax = plt.subplots(1,1,figsize=(18,10))
#     tree.plot_tree(random_forest_model.estimators_[i], feature_names = feature_names, ax=ax, fontsize=6, class_names=['No Disease', 'Has Disease'], filled=True)
#     filename = 'figures/random_forest/Fig_random_forest_tree_'+str(i+1)+'.png'
#     plt.show()

####### Logistic Regression #######
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

train_pred = classifier.predict(X_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print("Logistic Regression:Training accuracy: {:.2f}% | Test accuracy: {:.2f}% | F1 score: {:.3f}".format(accuracy_score(train_pred, y_train)*100, 
                                                                                  accuracy_score(y_pred, y_test)*100, 
                                                                                  f1_score(y_pred, y_test))) 

cm_test = confusion_matrix(y_pred, y_test)
cm_train = confusion_matrix(train_pred, y_train)
print("Train Confusion matrix", cm_train.ravel(), "| Test Confusion Matrix", cm_test.ravel())
acc.append(accuracy_score(y_pred, y_test)*100)
train_acc.append(accuracy_score(train_pred, y_train)*100)
F1s.append(f1_score(y_pred, y_test)*100)


####### SVM #######
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)

train_pred = classifier.predict(X_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print("SVM rbf:       Training accuracy: {:.2f}% | Test accuracy: {:.2f}% | F1 score: {:.3f}".format(accuracy_score(train_pred, y_train)*100, 
                                                                                  accuracy_score(y_pred, y_test)*100, 
                                                                                  f1_score(y_pred, y_test))) 

cm_test = confusion_matrix(y_pred, y_test)
cm_train = confusion_matrix(train_pred, y_train)
print("Train Confusion matrix", cm_train.ravel(), "| Test Confusion Matrix", cm_test.ravel())

classifier = SVC(kernel = 'poly')
classifier.fit(X_train, y_train)

train_pred = classifier.predict(X_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print("SVM poly:      Training accuracy: {:.2f}% | Test accuracy: {:.2f}% | F1 score: {:.3f}".format(accuracy_score(train_pred, y_train)*100, 
                                                                                  accuracy_score(y_pred, y_test)*100, 
                                                                                  f1_score(y_pred, y_test))) 
cm_test = confusion_matrix(y_pred, y_test)
cm_train = confusion_matrix(train_pred, y_train)
print("Train Confusion matrix", cm_train.ravel(), "| Test Confusion Matrix", cm_test.ravel())

classifier = SVC(kernel = 'linear')
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
acc.append(accuracy_score(y_pred, y_test)*100)
train_acc.append(accuracy_score(train_pred, y_train)*100)
F1s.append(f1_score(y_pred, y_test)*100)

####### Naive Bayes #######
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

train_pred = classifier.predict(X_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)
cm_train = confusion_matrix(train_pred, y_train)

print("Naive Bayes:   Training accuracy: {:.2f}% | Test accuracy: {:.2f}% | F1 score: {:.3f}".format(accuracy_score(train_pred, y_train)*100, 
                                                                                  accuracy_score(y_pred, y_test)*100, 
                                                                                  f1_score(y_pred, y_test))) 
print("Train Confusion matrix", cm_train.ravel(), "| Test Confusion Matrix", cm_test.ravel())
acc.append(accuracy_score(y_pred, y_test)*100)
train_acc.append(accuracy_score(train_pred, y_train)*100)
F1s.append(f1_score(y_pred, y_test)*100)


####### lightgbm #######
import lightgbm as lgb

d_train = lgb.Dataset(X_train, label = y_train)
params = {}

clf = lgb.train(params, d_train, 100)
#Prediction
y_pred = clf.predict(X_test)
#convert into binary values
for i in range(0, len(y_pred)):
    if y_pred[i]>= 0.5:       # setting threshold to .5
       y_pred[i]=1
    else:  
       y_pred[i]=0
       
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = clf.predict(X_train)

for i in range(0, len(y_pred_train)):
    if y_pred_train[i]>= 0.5:       # setting threshold to .5
       y_pred_train[i]=1
    else:  
       y_pred_train[i]=0
       
cm_train = confusion_matrix(y_pred_train, y_train)

print("lightgbm:      Training accuracy: {:.2f}% | Test accuracy: {:.2f}% | F1 score: {:.3f}".format(accuracy_score(y_pred_train, y_train)*100, 
                                                                                  accuracy_score(y_pred, y_test)*100, 
                                                                                  f1_score(y_pred, y_test))) 
print("Train Confusion matrix", cm_train.ravel(), "| Test Confusion Matrix", cm_test.ravel())
acc.append(accuracy_score(y_pred, y_test)*100)
train_acc.append(accuracy_score(y_pred_train, y_train)*100)
F1s.append(f1_score(y_pred, y_test)*100)


####### XGBoost #######
from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = xg.predict(X_train)

for i in range(0, len(y_pred_train)):
    if y_pred_train[i]>= 0.5:       # setting threshold to .5
       y_pred_train[i]=1
    else:  
       y_pred_train[i]=0
       
cm_train = confusion_matrix(y_pred_train, y_train)
print("XGBoost:     Training accuracy: {:.2f}% | Test accuracy: {:.2f}% | F1 score: {:.3f}".format(accuracy_score(y_pred_train, y_train)*100, 
                                                                                  accuracy_score(y_pred, y_test)*100, 
                                                                                  f1_score(y_pred, y_test))) 
print("Train Confusion matrix", cm_train.ravel(), "| Test Confusion Matrix", cm_test.ravel())
acc.append(accuracy_score(y_pred, y_test)*100)
train_acc.append(accuracy_score(y_pred_train, y_train)*100)
F1s.append(f1_score(y_pred, y_test)*100)


####### AdaBoost #######
from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

train_pred = classifier.predict(X_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)
cm_train = confusion_matrix(train_pred, y_train)

print("AdaBoost 20:     Training accuracy: {:.2f}% | Test accuracy: {:.2f}% | F1 score: {:.3f}".format(accuracy_score(train_pred, y_train)*100, 
                                                                                  accuracy_score(y_pred, y_test)*100, 
                                                                                  f1_score(y_pred, y_test))) 
print("Train Confusion matrix", cm_train.ravel(), "| Test Confusion Matrix", cm_test.ravel())
acc.append(accuracy_score(y_pred, y_test)*100)
train_acc.append(accuracy_score(train_pred, y_train)*100)
F1s.append(f1_score(y_pred, y_test)*100)

####### Multi-layer Perceptron #######
# hidden_layer_sizes=(100,), activation='relu', *, solver='adam', 
# alpha=0.0001, batch_size='auto', learning_rate='constant', 
# learning_rate_init=0.001, power_t=0.5, max_iter=200
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(random_state=1, max_iter=300, 
                           hidden_layer_sizes=(25, 50), 
                           activation='logistic',
                           learning_rate_init=0.001,
                           solver='adam')
classifier.fit(X_train, y_train)

train_pred = classifier.predict(X_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_pred, y_test)
cm_train = confusion_matrix(train_pred, y_train)

print("Multi-layer Perceptron:Training accuracy: {:.2f}% | Test accuracy: {:.2f}% | F1 score: {:.3f}".format(accuracy_score(train_pred, y_train)*100, 
                                                                                  accuracy_score(y_pred, y_test)*100, 
                                                                                  f1_score(y_pred, y_test))) 
print("Train Confusion matrix", cm_train.ravel(), "| Test Confusion Matrix", cm_test.ravel())
acc.append(accuracy_score(y_pred, y_test)*100)
train_acc.append(accuracy_score(train_pred, y_train)*100)
F1s.append(f1_score(y_pred, y_test)*100)

# Visualize all results
models = ['SVM (Linear)', 'Naive Bayes', 'Logistic Reg', 'Decision Tree', 'Random Forest', 'LightGBM', 'XGBoost', 'AdaBoost', 
          'MLP']

# set width of bar 
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
barWidth = 0.25

# Set position of bar on X axis 
br1 = np.arange(len(train_acc)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 

rect1 = ax.bar(br1, train_acc, width = barWidth, label = 'Train Accuracy', linewidth = 2, color = [224/250, 201/255, 191/255, 1])
ax.bar_label(rect1, np.round(train_acc, 1), fmt='%.2f', padding = 3)
rect2 = ax.bar(br2, acc, width = barWidth, label = 'Test Accuracy', linewidth = 2, color = [191/250, 156/255, 141/255, 1])
ax.bar_label(rect2, np.round(acc, 1), fmt='%.2f', padding = 3)
rect3 = ax.bar(br3, F1s, width = barWidth, label = 'F1 score', linewidth = 2, color = [64/250, 99/255, 144/255, 1])
ax.bar_label(rect3, np.round(F1s, 1), fmt='%.2f', padding = 3)

plt.xticks([r + barWidth for r in range(len(train_acc))], 
        models)

plt.legend(loc = 'best')

plt.ylim([80, 102])
plt.yticks([80 + i * 5 for i in range(5)])
plt.ylabel("Accuracy [%]/F1 score")
plt.show()