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
    

    # mf = MetricFrame(metrics=accuracy_score, y_true=y_test, y_pred=test_pred, sensitive_features=A_test)
    # print("Overall Accuracy: ", mf.overall)
    # print(mf.by_group)

    # sr = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=test_pred, sensitive_features=A_test)
    # print("Selection Rate: ", sr.overall)
    # print(sr.by_group)

    metrics = {
        # "demographic_parity_ratio": demographic_parity_ratio,
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "balanced_acc_error": balanced_accuracy_error,
        "selection_rate": selection_rate,
        "false_positive_rate": false_positive_rate,
        "false_positive_error": false_positive_error,
        "false_negative_rate": false_negative_rate,
        "false_negative_error": false_negative_error,
    }

    metric_frame = MetricFrame(
        metrics=metrics, y_true=y_test, y_pred=test_pred, sensitive_features=A_test
    )
    print(metric_frame.overall)
    print(metric_frame.by_group)

    print("Equalized odds ratio: ", equalized_odds_ratio(y_true=y_test, y_pred=test_pred, sensitive_features=A_test))
    print("Demographic Parity Ratio: ", demographic_parity_ratio(y_true=y_test, y_pred=test_pred, sensitive_features=A_test))

    ax = metric_frame.by_group.plot.bar(
        subplots=True,
        layout=[3, 3],
        legend=False,
        figsize=[15, 9],
        title="Different Metrics used on Model",
    )

    # x_offset = -0.12
    # y_offset = -0.005
    # for i in range(3):
    #     for j in range(3):
    #         for p in ax[i, j].patches:
    #             b = p.get_bbox()
    #             val = "{:.2f}".format(b.y1 + b.y0) 
    #             if b.y1 + b.y0 == 0:       
    #                 ax[i, j].annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 - y_offset))
    #             else:
    #                 ax[i, j].annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))
    # plt.savefig("figures/fairness/Fig_"+model_name+"_metrics.png")
    # plt.show()

    postprocess_est = ThresholdOptimizer(
    estimator = model,
    constraints="selection_rate_parity",  # Optimize FPR and FNR simultaneously
    objective="balanced_accuracy_score",
    prefit=True,
    predict_method="predict_proba",)
    
    postprocess_est.fit(X=X_train, y=y_train, sensitive_features=A_train)

    postprocess_pred = postprocess_est.predict(X_test, sensitive_features=A_test)

    postprocess_pred_proba = postprocess_est._pmf_predict(
        X_test, sensitive_features=A_test
    )

    if model_name == 'LightGBM':
        for i in range(0, len(postprocess_pred)):
            if postprocess_pred[i]>= 0.5:       # setting threshold to .5
                postprocess_pred[i]=1
            else:  
                postprocess_pred[i]=0

    print("======Post Processing======")
    mf = MetricFrame(metrics=accuracy_score, y_true=y_test, y_pred=postprocess_pred, sensitive_features=A_test)
    print("Overall Accuracy: ", mf.overall)
    print(mf.by_group)

    sr = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=postprocess_pred, sensitive_features=A_test)
    print("Selection Rate: ", sr.overall)
    print(sr.by_group)

    bal_acc_postprocess = balanced_accuracy_score(y_test, postprocess_pred)
    eq_odds_postprocess = equalized_odds_difference(
    y_test, postprocess_pred, sensitive_features=A_test)

    metricframe_postprocess = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=postprocess_pred,
        sensitive_features=A_test,
    )

    metrics_to_report = [
    "balanced_accuracy",
    "false_positive_rate",
    "false_negative_rate",
    "selection_rate"]

    metricframe_postprocess.overall[metrics_to_report]
    print(metricframe_postprocess.difference()[metrics_to_report])

    metricframe_cmp = compare_metricframe_results(
    metric_frame,
    metricframe_postprocess,
    metrics=metrics_to_report,
    names=["Unmitigated", "PostProcess"],
    )
    
    metricframe_cmp.plot.bar(subplots=True, figsize=[16, 8], layout=[4, 2], legend=None, title='Compare unmitigated and mitigated')
    # metricframe_postprocess.by_group[metrics_to_report].plot.bar(
    #     subplots=True, layout=[1, 4], figsize=[12, 4], legend=None, rot=0
    # )


    # plt.show()