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
X = pd.get_dummies(df.drop(columns=["HeartDisease"]), dtype = float)

A_str = A.map({1: "male", 2: "female"})
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X, Y, A_str, test_size=0.2, stratify=Y
)

X_train, y_train, A_train = f.resample_training_data(X_train, y_train, A_train)

features = [2, 3, 5]

########################
#     Decision Tree    #
########################

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.naive_bayes import GaussianNB
classifier = DecisionTreeClassifier(min_samples_split = 50, max_depth=3, random_state=21)

clf = classifier.fit(X_train, y_train)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
pdp_plot = PartialDependenceDisplay.from_estimator(clf, X_train, features, ax = ax, 
                                                   line_kw={"label": "Decision Tree"})
plt.tight_layout()
plt.savefig("figures/PDP_plots/Fig_DecisionTree_PDP.png")
plt.show()


########################
# Gaussian Naive Bayes #
########################

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

clf = classifier.fit(X_train, y_train)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
pdp_plot = PartialDependenceDisplay.from_estimator(clf, X_train, features, ax = ax, 
                                                   line_kw={"label": "Naive Bayes"})
plt.tight_layout()
plt.savefig("figures/PDP_plots/Fig_GaussianNB_PDP.png")
plt.show()



########################
#    Random Forest     #
########################
random_forest_model = RandomForestClassifier(max_depth = 8, 
                                            min_samples_split = 100,
                                            n_estimators = 10,
                                            random_state = 21).fit(X_train, y_train)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
pdp_plot = PartialDependenceDisplay.from_estimator(random_forest_model, X_train, 
                                                   features, ax = ax, 
                                                   line_kw={"label": "Random Forest", "color": "red"})
plt.tight_layout()
plt.savefig("figures/PDP_plots/Fig_rf_PDP.png")

plt.show()

#########################
#  Logistic Regression  #
#########################

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

clf = classifier.fit(X_train, y_train)

# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
pdp_plot = PartialDependenceDisplay.from_estimator(clf, X_train, features, ax = ax, 
                                                   line_kw={"label": "Logistic Regression", "color":"green"})
plt.tight_layout()
plt.savefig("figures/PDP_plots/Fig_logistic_regression_PDP.png")
plt.show()

########################
#       AdaBoost       #
########################
from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier(n_estimators=20).fit(X_train, y_train)

#fig, ax = plt.subplots(1, 3, figsize=(12, 4))
pdp_plot = PartialDependenceDisplay.from_estimator(classifier, X_train, features, ax = ax, 
                                                   line_kw={"label": "Adaboost", "color": "green"})
plt.tight_layout()
plt.savefig("figures/PDP_plots/Fig_adaboost_PDP.png")

plt.show()


########################
#Multilayer  Perceptron#
########################
from sklearn.neural_network import MLPClassifier
classifier = MLPClassifier(random_state=1, max_iter=300, 
                           hidden_layer_sizes=(25, 50), 
                           activation='logistic',
                           learning_rate_init=0.001,
                           solver='adam').fit(X_train, y_train)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
pdp_plot = PartialDependenceDisplay.from_estimator(classifier, X_train, features, ax = ax, 
                                                   line_kw={"label": "MLP", "color":"magenta"})
plt.tight_layout()
plt.savefig("figures/PDP_plots/Fig_mlp_PDP.png")
plt.show()

########################
#        XGBoost       #
########################
from xgboost import XGBClassifier
xg = XGBClassifier()
xg.fit(X_train, y_train)
#Prediction

pdp_plot = PartialDependenceDisplay.from_estimator(xg, X_train, features, ax = ax, 
                                                   line_kw={"label": "XGboost", "color": "magenta"})
plt.tight_layout()
ax[0].legend(loc = 'best')
plt.savefig("figures/PDP_plots/Fig_XGBoost_PDP.png")
# ax[1].legend(loc = 'best')
# ax[2].legend(loc = 'best')
plt.show()

########################
#       LightGBM       #
########################
import lightgbm as lgb # need to return to version 4.1.0

# params = {}

# d_train = lgb.Dataset(X_train, label = y_train)
# lgbm_sk = lgb.LGBMRegressor(metric='l1', objective='regression',deterministic=True, random_state=4)

# lgbm_sk.fit(X_train.to_numpy(), y_train.to_numpy())


# #Prediction
# features = [0, 1, (0, 1)]

# fig, ax = plt.subplots(1, 3, figsize=(12, 4))
# pdp_plot = PartialDependenceDisplay.from_estimator(lgbm_sk, X_train, features, ax = ax)
# plt.tight_layout()
# #plt.savefig("figures/PDP_plots/Fig_lightgbm_PDP.png")
# plt.show()

########################
#Support Vector Machine#
########################
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', probability=True).fit(X_train, y_train)
# features = [0, 1, (0, 1)]

# fig, ax = plt.subplots(1, 3, figsize=(12, 4))
pdp_plot = PartialDependenceDisplay.from_estimator(classifier, X_train, features, ax = ax, 
                                                   line_kw={"label": "SVM", "color":"red"})
plt.tight_layout()
plt.savefig("figures/PDP_plots/Fig_SVM_PDP.png")
plt.show()