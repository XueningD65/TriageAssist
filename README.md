# TriageAssist - Auditing

## This repository is cloned from [Heart Disease Prediction](https://github.com/ShubhankarRawat/Heart-Disease-Prediction.git)

## Introduction

Cardiovascular diseases (CVDs) are the leading cause of death worldwide, resulting in 17.9 million deaths annually, or 31% of global deaths. In a major hospital's emergency department, swift and accurate triaging can be the difference between life and death, especially for patients with potential heart conditions. 
Using machine learning, we are able to detect the common patterns between CVD patients and predict whether the patient is at high risk.


## Dataset Attribute Information
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

## Dataset Source

   - Cleveland: 303 observations
   - Hungarian: 294 observations
   - Switzerland: 123 observations
   - Long Beach VA: 200 observations
   - Stalog (Heart) Data Set: 270 observations

## Model Training and Prediction : 
We can train our prediction model by analyzing existing data because we already know whether each patient has heart disease. This process is also known as supervision and learning. The trained model is then used to predict if users suffer from heart disease. The training and prediction process is described as follows:

## Splitting: 
First, data is divided into two parts using component splitting. In this experiment, data is split based on a ratio of 80:20 for the training set and the prediction set. The training set data is used in the logistic regression component for model training, while the prediction set data is used in the prediction component.

The following classification models are used - Logistic Regression, Random Forest Classfier, SVM, Naive Bayes Classifier, Decision Tree Classifier, LightGBM, XGBoost

## Prediction:
The two inputs of the prediction component are the model and the prediction set. The prediction result shows the predicted data, actual data, and the probability of different results in each group.

## Evaluation: 
The confusion matrix, also known as the error matrix, is used to evaluate the accuracy of the model.
