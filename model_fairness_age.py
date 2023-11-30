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
import seaborn as sns

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
    true_positive_rate,
    true_negative_rate,
    equalized_odds_ratio,
    demographic_parity_ratio,
    
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

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

import helper_func as f
import argparse
import sys


def compute_error_metric(metric_value, sample_size):
    """Compute standard error of a given metric based on the assumption of
    normal distribution.

    Parameters:
    metric_value: Value of the metric
    sample_size: Number of data points associated with the metric

    Returns:
    The standard error of the metric
    """
    metric_value = metric_value / sample_size
    return (
        1.96
        * np.sqrt(metric_value * (1.0 - metric_value))
        / np.sqrt(sample_size)
    )


def false_positive_error(y_true, y_pred):
    """Compute the standard error for the false positive rate estimate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return compute_error_metric(fp, tn + fp)


def false_negative_error(y_true, y_pred):
    """Compute the standard error for the false negative rate estimate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return compute_error_metric(fn, fn + tp)


def balanced_accuracy_error(y_true, y_pred):
    """Compute the standard error for the balanced accuracy estimate."""
    fpr_error, fnr_error = false_positive_error(
        y_true, y_pred
    ), false_negative_error(y_true, y_pred)
    return np.sqrt(fnr_error**2 + fpr_error**2) / 2

def compare_metricframe_results(mframe_1, mframe_2, metrics, names):
    """Concatenate the results of two MetricFrames along a subset of metrics.

    Parameters
    ----------
    mframe_1: First MetricFrame for comparison
    mframe_2: Second MetricFrame for comparison
    metrics: The subset of metrics for comparison
    names: The names of the selected metrics

    Returns
    -------
    MetricFrame : MetricFrame
        The concatenation of the two MetricFrames, restricted to the metrics
        specified.

    """
    return pd.concat(
        [mframe_1.by_group[metrics], mframe_2.by_group[metrics]],
        keys=names,
        axis=1,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("square", help="INPUT FROM: Decision Tree = DT, Random Forest = RF, Logistic Reg = LR, SVM (Linear), Naive Bayes = NB, LightGBM, XGBoost = XGB, AdaBoost, MLP",
                type=int)

    ####### Load Dataset #######
    df = pd.read_csv('heart.csv', header = 0)
    df.columns = ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope','HeartDisease']

    df['Sex'] = df.Sex.map({'F': 2, 'M': 1})
    Y, A = df.loc[:, "HeartDisease"], df.loc[:, "Sex"]
    B = df.loc[:, "Age"]
    X = pd.get_dummies(df.drop(columns=["HeartDisease"]), dtype = float)

    A_str = A.map({1: "male", 2: "female"})
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, Y, B, test_size=0.2, stratify=Y
    )
    X_train, y_train, A_train = f.resample_training_data(X_train, y_train, A_train)

    ####### Load the models #######

    model_name = sys.argv[1]

    if model_name == 'DT':
        model = DecisionTreeClassifier(min_samples_split = 50, max_depth=3, random_state=21)
    elif model_name == 'RF':
        model = RandomForestClassifier(max_depth = 8, 
                                            min_samples_split = 100,
                                            n_estimators = 10,
                                            random_state = 21)
    elif model_name == 'SVM':
        model = SVC(kernel = 'linear', probability=True)
    elif model_name == 'NB':
        model = GaussianNB()
    elif model_name == 'LR':
        model = LogisticRegression()
    elif model_name == 'XGB':
        model = XGBClassifier()
    elif model_name == 'Adaboost':
        model = AdaBoostClassifier(n_estimators=20)
    elif model_name == 'MLP':
        model = MLPClassifier(random_state=1, max_iter=300, 
                           hidden_layer_sizes=(25, 50), 
                           activation='logistic',
                           learning_rate_init=0.001,
                           solver='adam')
    elif model_name == 'LightGBM':
        d_train = lgb.Dataset(X_train, label = y_train)
        params = {}
        model = lgb.train(params, d_train, 100)

    feature_names = list(X.columns)

    if model_name != "LightGBM":
        model = model.fit(X_train, y_train)

    ####### Training and testing of the model #######
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    if model_name == 'LightGBM':
        #Prediction
        for i in range(0, len(train_pred)):
            if train_pred[i]>= 0.5:       # setting threshold to .5
                train_pred[i]=1
            else:  
                train_pred[i]=0

        for i in range(0, len(test_pred)):
            if test_pred[i]>= 0.5:       # setting threshold to .5
                test_pred[i]=1
            else:  
                test_pred[i]=0
       

    print("Model: Training accuracy: {:.2f}% | Test accuracy: {:.2f}% | F1 score: {:.3f}".format(accuracy_score(train_pred, y_train)*100, 
                                                                                    accuracy_score(test_pred, y_test)*100, 
                                                                                    f1_score(test_pred, y_test))) 
    
    mf = MetricFrame(metrics=accuracy_score, y_true=y_test, y_pred=test_pred, sensitive_features=A_test)
    print("Overall Accuracy: ", mf.overall)

    ####### Divide the numerical ages into groups #######

    age_group = mf.by_group._data.items.to_numpy()
    accuracy_age_group = mf.by_group.to_numpy()

    age_count = X_test.loc[:, "Age"].value_counts()
    age_idx = age_count._data.items.to_numpy()
    age_frequency = age_count.to_numpy()
    age_freq_dict = {}
    age_positive_dict = {"39-":0, "40-49":0, "50-59":0, "60-69":0, "70+":0}

    for i, age in enumerate(age_idx):
        age_freq_dict[age] = age_frequency[i]

    age_acc_dict = {"39-":0, "40-49":0, "50-59":0, "60-69":0, "70+":0}
    age_freq_dict_group = {"39-":0, "40-49":0, "50-59":0, "60-69":0, "70+":0}
    for i, age in enumerate(age_group):
        if age < 40:
            age_acc_dict["39-"] += accuracy_age_group[i] * age_freq_dict[age]
            age_freq_dict_group["39-"] += age_freq_dict[age]

        elif age < 50:
            age_acc_dict["40-49"] += accuracy_age_group[i] * age_freq_dict[age]
            age_freq_dict_group["40-49"] += age_freq_dict[age]

        elif age < 60:
            age_acc_dict["50-59"] += accuracy_age_group[i] * age_freq_dict[age]
            age_freq_dict_group["50-59"] += age_freq_dict[age]

        elif age < 70:
            age_acc_dict["60-69"] += accuracy_age_group[i] * age_freq_dict[age]
            age_freq_dict_group["60-69"] += age_freq_dict[age]

        else:
            age_acc_dict["70+"] += accuracy_age_group[i] * age_freq_dict[age]
            age_freq_dict_group["70+"] += age_freq_dict[age]
    
    for idx in age_freq_dict_group.keys():
        age_acc_dict[idx] = age_acc_dict[idx] * 100 / age_freq_dict_group[idx]

    for i, age in enumerate(X_train.loc[:, "Age"]):
        if age < 40:
            if y_train.to_numpy()[i] == 1:
                age_positive_dict["39-"] += 1
        elif age < 50:
            if y_train.to_numpy()[i] == 1:
                age_positive_dict["40-49"] += 1
        elif age < 60:
            if y_train.to_numpy()[i] == 1:
                age_positive_dict["50-59"] += 1
        elif age < 70:
            if y_train.to_numpy()[i] == 1:
                age_positive_dict["60-69"] += 1
        else:
            if y_train.to_numpy()[i] == 1:
                age_positive_dict["70+"] += 1

    print(age_positive_dict)
    print(age_freq_dict_group)

    B_str = B.apply(lambda age: "39-" if age < 40 else "40-49" if age < 50 else "50-59" if age < 60 else "60-69" if age < 70 else "70+")

    A_test = A_test.apply(lambda age: "39-" if age < 40 else "40-49" if age < 50 else "50-59" if age < 60 else "60-69" if age < 70 else "70+")
    A_train = A_train.apply(lambda age: "39-" if age < 40 else "40-49" if age < 50 else "50-59" if age < 60 else "60-69" if age < 70 else "70+")
    print(A_train.value_counts())

    ####### Compute and print selective metrics #######
    
    mf = MetricFrame(metrics=accuracy_score, y_true=y_test, y_pred=test_pred, sensitive_features=A_test)
    print("Overall Accuracy: ", mf.overall)
    print(mf.by_group)

    mf = MetricFrame(metrics=true_positive_rate, y_true=y_test, y_pred=test_pred, sensitive_features=A_test)
    print("Overall Sensitivity: ", mf.overall)
    print(mf.by_group)

    mf = MetricFrame(metrics=true_negative_rate, y_true=y_test, y_pred=test_pred, sensitive_features=A_test)
    print("Overall Specificity: ", mf.overall)
    print(mf.by_group)

    print("Equalized odds ratio: ", equalized_odds_ratio(y_true=y_test, y_pred=test_pred, sensitive_features=A_test))
    print("Equalized odds difference: ", equalized_odds_difference(y_true=y_test, y_pred=test_pred, sensitive_features=A_test))

    # fig, ax = plt.subplots(1, 1, figsize = (8, 4))
    # ax.plot(age_acc_dict.keys(), age_acc_dict.values())
    # plt.show()

