import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Visualize all results
models = ['Decision Tree', 'Random Forest', 'LightGBM', 'XGBoost', 'AdaBoost', 'SVM (Linear)', 'Naive Bayes', 'Logistic Reg', 'MLP']
total_acc = [85.33, 87.50, 85.87, 85.87, 88.59, 87.50, 88.04, 88.59, 85.87]
male_acc = [85.42, 88.89, 89.58, 89.58, 88.89, 86.11, 87.50, 87.50, 83.33]
female_acc = [85.00, 82.50, 72.50, 72.50, 87.50, 92.50, 90.00, 92.50, 95.00]

# set width of bar 
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
barWidth = 0.25

# Set position of bar on X axis 
br1 = np.arange(len(total_acc)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 

rect1 = ax.bar(br1, total_acc, width = barWidth, label = 'Total Accuracy', linewidth = 2, color = [192/250, 179/255, 210/255, 1])
ax.bar_label(rect1, np.round(total_acc, 1), fmt='%.2f', padding = 3)
rect2 = ax.bar(br2, male_acc, width = barWidth, label = 'Male Accuracy', linewidth = 2, color = [212/250, 151/255, 157/255, 1])
ax.bar_label(rect2, np.round(male_acc, 1), fmt='%.2f', padding = 3)
rect3 = ax.bar(br3, female_acc, width = barWidth, label = 'Female Accuracy', linewidth = 2, color = [142/250, 206/255, 233/255, 1])
ax.bar_label(rect3, np.round(female_acc, 1), fmt='%.2f', padding = 3)

plt.xticks([r + barWidth for r in range(len(total_acc))], 
        models)

plt.legend(loc = 'upper left')

plt.ylim([70, 98])
plt.yticks([70 + i * 5 for i in range(6)])
plt.ylabel("Accuracy [%]")
plt.savefig("figures/accuracy/Fig_acc_with_gender.png", dpi=300)
plt.show()

total_acc = [0.912, 0.931, 0.902, 0.922, 0.892, 0.882, 0.882, 0.892, 0.824]
male_acc = [0.904, 0.936, 0.915, 0.947, 0.894, 0.883, 0.883, 0.894, 0.809]
female_acc = [1.000, 0.875, 0.750, 0.625, 0.875, 0.875, 0.875, 0.875, 1.000]

# set width of bar 
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
barWidth = 0.25

# Set position of bar on X axis 
br1 = np.arange(len(total_acc)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 

rect1 = ax.bar(br1, total_acc, width = barWidth, label = 'Total Sensitivity', linewidth = 2, color = [192/250, 179/255, 210/255, 1])
ax.bar_label(rect1, np.round(total_acc, 2), fmt='%.3f', padding = 3)
rect2 = ax.bar(br2, male_acc, width = barWidth, label = 'Male Sensitivity', linewidth = 2, color = [212/250, 151/255, 157/255, 1])
ax.bar_label(rect2, np.round(male_acc, 2), fmt='%.3f', padding = 3)
rect3 = ax.bar(br3, female_acc, width = barWidth, label = 'Female Sensitivity', linewidth = 2, color = [142/250, 206/255, 233/255, 1])
ax.bar_label(rect3, np.round(female_acc, 2), fmt='%.3f', padding = 3)

plt.xticks([r + barWidth for r in range(len(total_acc))], 
        models)

plt.legend(loc = 'best')

plt.ylim([0.60, 1.02])
plt.yticks([0.60 + i * 0.1 for i in range(5)])
plt.ylabel("Sensitivity")
plt.savefig("figures/accuracy/Fig_sensitivity_with_gender.png", dpi=300)
plt.show()