# TriageAssist - Auditing with Quantitative Analysis

## This repository is adapted from [Heart Disease Prediction](https://github.com/ShubhankarRawat/Heart-Disease-Prediction.git)

## Introduction

Cardiovascular diseases (CVDs) are the leading cause of death worldwide, resulting in 17.9 million deaths annually, or 31% of global deaths. In a major hospital's emergency department, swift and accurate triaging can be the difference between life and death, especially for patients with potential heart conditions. 

Triaging in emergency departments describes a process of scoring and sorting the patients according to their level of urgency to optimize the distribution of medical resources. The TriageAssist to be developed in this repository uses machine learning algorithms to find out the common patterns in cardivascular diseases symptoms and predict the probability of heart diseases subject to the patient data. Such system is expected to act as an additional indication to the healthcare professions for faster and more accurate triaging process.

Despite the anticipated convenience brought by the TriageAssist system, problems can arise when the system is not properly developed and its prediction is not properly understood. Therefore, in this repository, we apply model interpretation and fairness investigation toolkits to quantitatively analyze a range of machine learning algorithms. Our objective is to spot the common pitfalls in these models and alarm the potential stakeholders about them.


## Install packages
```
pip install -r requirements.txt
```

## Dataset Attribute Information
The dataset is stored in the [heart.csv](../main/heart.csv) file.
   1. Age: age of the patient [years]
   2. Sex: sex of the patient [M: Male, F: Female]
   3. ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
   4. RestingBP: resting blood pressure [mm Hg]
   5. Cholesterol: serum cholesterol [mm/dl]
   6. FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
   7. RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
   8. MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
   9. ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
   10. Oldpeak: oldpeak = ST [Numeric value measured in depression]
   11. ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
   12. HeartDisease: output class [1: heart disease, 0: Normal]

## Dataset Description

   - Cleveland: 303 observations
   - Hungarian: 294 observations
   - Switzerland: 123 observations
   - Long Beach VA: 200 observations
   - Stalog (Heart) Data Set: 270 observations

   The dataset is split into training and test set by the ratio 8:2. The training set has been augmented to ensure balance between the genders.

   To visualize the distribution of the dataset, please go through [dataset_visualize.ipynb](../main/dataset_visualize.ipynb) by running cell by cell.

## Models

1. **Tree-based Classifiers**:
   
   Decision Tree (DT), Random Forest (RF), AdaBoost, XGBoost (XGB), LightGBM.
   
2. **Other statistical Classifiers**:
   
   Gaussian Naive Bayes (NB), Support Vector Machine (SVM), Logistic Regression (LR), Multi-Layer Perceptron (MLP).

### Model Training and Prediction
[helper_func.py](../main/helper_func.py) contains a function which balances the number of labels.
To see model training results before balancing:
Run it directly by
```
python3 train_without_balance.py
```
To see model training results after balancing:
Run it directly by
```
python3 train.py
```
Both files contain all the models trained and their results, including training and test accuracy, F1 score and confusion matrix. The train set and test set are split by 8:2.

## Interpret Black Box Models

### Feature Importance (FI)
`SHAP` is used for interpreting the black box models with feature importance (FI).
For those directly plotted in `matplotlib`, run
```
python3 model_explain_FI_SHAP.py
```

For those requiring HTML to display results, open [model_explain_FI_HTML.ipynb](../main/model_explain_FI_HTML.ipynb) and run cell by cell.

We also verified the ranking of feature importance, and whether they positively or negatively impacted the final decision using `LIME` package. To see the results from it, open [model_explain_FI_LIME.ipynb](../main/model_explain_FI_LIME.ipynb) and run cell by cell.

### Partial Dependence Plot (PDP)
`SHAP` and `sklearn` is used for interpreting the black box models with feature importance (FI).
For those directly plotted in `matplotlib`, run
```
python3 model_explain_PDP.py
```
The figures will be automatically saved.

## Audit the fairness of models
`Fairlearn` is used for measureing all types of fairness metrics, to view all kinds of results, run
```
python3 model_fairness_gender.py [model_name_abbr]
python3 model_fariness_age.py [model_name_abbr]
```
where models include Decision Tree [DT], Random Forest [RF], Logistic Reg [LR], [SVM] (Linear), Naive Bayes [NB], [LightGBM], XGBoost [XGB], [Adaboost], multi-layer perceptron [MLP]. Use the names in [] when running the program.

[dataset_load_and_test.ipynb](../main/dataset_load_and_test.ipynb) also contains a brief walkthrough of the Fairlearn toolkits. To see the result, open it and run cell by cell.

## Result Visualization
To visualize the results from audits, run
```
python3 plots.py
```
The figures are automatically saved in the folders.
