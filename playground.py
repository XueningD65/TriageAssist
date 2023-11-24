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
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    X = pd.get_dummies(df.drop(columns=["HeartDisease"]), dtype = float)

    A_str = A.map({1: "male", 2: "female"})
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, Y, A_str, test_size=0.2, stratify=Y
    )
    X_train, y_train, A_train = f.resample_training_data(X_train, y_train, A_train)

    # print("=======Dataset Information========")
    # print(A_str.value_counts())
    # print(A_train.value_counts())
    # print(A_test.value_counts())

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

    from xgboost import XGBClassifier
    xg = XGBClassifier(n_estimators=100,
                                 max_depth=3,  # limiting depth of trees
                                 learning_rate=0.1,  # potentially adding regularization via learning rate
                                 subsample=0.8,  # using a subsample of data to prevent overfitting
                                 colsample_bytree=0.7,  # using a subsample of features for each tree
                                 eval_metric='logloss',
                                 random_state=42)
    xg.fit(X_train, y_train)
    y_pred = xg.predict(X_test)

    from sklearn.metrics import confusion_matrix
    cm_test = confusion_matrix(y_pred, y_test)

    y_pred_train = xg.predict(X_train)
        
    cm_train = confusion_matrix(y_pred_train, y_train)
    print("XGBoost:     Training accuracy: {:.2f}% | Test accuracy: {:.2f}% | F1 score: {:.3f}".format(accuracy_score(y_pred_train, y_train)*100, 
                                                                                    accuracy_score(y_pred, y_test)*100, 
                                                                                    f1_score(y_pred, y_test))) 
    print("Train Confusion matrix", cm_train.ravel(), "| Test Confusion Matrix", cm_test.ravel())

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=xg.classes_)
    disp.plot(text_kw ={"fontsize":15})
   # plt.savefig("figures/confusion_matrix/Fig_cm_xgboost.png")
    plt.show()

    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5)
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

    # Recreate the stacked model with base models
    model_stack = StackingClassifier(estimators=[('rf', random_forest_model),
                                                ('xgb', xg),
                                                ('knn', classifier)],
                                            final_estimator=LogisticRegression(),
                                            stack_method='auto',
                                            n_jobs=-1)
    model_stack.fit(X_train, y_train)
    y_pred = model_stack.predict(X_test)
    cm_test = confusion_matrix(y_pred, y_test)
    y_pred_train = xg.predict(X_train)
        
    cm_train = confusion_matrix(y_pred_train, y_train)
    print("Stacking: Training accuracy: {:.2f}% | Test accuracy: {:.2f}% | F1 score: {:.3f}".format(accuracy_score(y_pred_train, y_train)*100, 
                                                                                    accuracy_score(y_pred, y_test)*100, 
                                                                                    f1_score(y_pred, y_test))) 
    print("Train Confusion matrix", cm_train.ravel(), "| Test Confusion Matrix", cm_test.ravel())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=xg.classes_)
    disp.plot(text_kw ={"fontsize":15})
   # plt.savefig("figures/confusion_matrix/Fig_cm_xgboost.png")
    plt.show()
