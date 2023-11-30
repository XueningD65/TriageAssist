import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('axes', titlesize=10)     # fontsize of the axes title
plt.rc('axes', labelsize=10)    # fontsize of the x and y labels

# Visualize all results
models = ['Decision Tree', 'Random Forest', 'LightGBM', 'XGBoost', 'AdaBoost', 'SVM (Linear)', 'Naive Bayes', 'Logistic Reg', 'MLP']
total_acc = [85.33, 87.50, 85.87, 85.87, 88.59, 87.50, 88.04, 88.59, 85.87]
male_acc = [85.42, 88.89, 89.58, 89.58, 88.89, 86.11, 87.50, 87.50, 83.33]
female_acc = [85.00, 82.50, 72.50, 72.50, 87.50, 92.50, 90.00, 92.50, 95.00]

barWidth = 0.2
# set width of bar 
fig, ax = plt.subplots(1, 1, figsize=(15, 5))

# Set position of bar on X axis 
br1 = np.arange(len(total_acc)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 

rect1 = ax.bar(br1, total_acc, width = barWidth, label = 'Total Accuracy', linewidth = 2, color = '#EAAA60')
ax.bar_label(rect1, np.round(total_acc, 1), fmt='%.2f', padding = 3)
rect2 = ax.bar(br2, male_acc, width = barWidth, label = 'Male Accuracy', linewidth = 2, color = '#E68B81')
ax.bar_label(rect2, np.round(male_acc, 1), fmt='%.2f', padding = 3)
rect3 = ax.bar(br3, female_acc, width = barWidth, label = 'Female Accuracy', linewidth = 2, color = '#B7B2D0')
ax.bar_label(rect3, np.round(female_acc, 1), fmt='%.2f', padding = 3)

plt.xticks([r + barWidth for r in range(len(total_acc))], 
        models)

plt.legend(loc = 'best', fontsize="12")

plt.ylim([70, 98])
plt.yticks([70 + i * 5 for i in range(6)])
plt.ylabel("Accuracy [%]")
plt.savefig("figures/accuracy/Fig_acc_with_gender.png", dpi=300)
plt.show()

total_acc = [0.912, 0.931, 0.902, 0.922, 0.892, 0.882, 0.882, 0.892, 0.824]
male_acc = [0.904, 0.936, 0.915, 0.947, 0.894, 0.883, 0.883, 0.894, 0.809]
female_acc = [1.000, 0.875, 0.750, 0.625, 0.875, 0.875, 0.875, 0.875, 1.000]

# set width of bar 
fig, ax = plt.subplots(1, 1, figsize=(15, 5))

# Set position of bar on X axis 
br1 = np.arange(len(total_acc)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 

rect1 = ax.bar(br1, total_acc, width = barWidth, label = 'Total Sensitivity', linewidth = 2, color = '#EAAA60')
ax.bar_label(rect1, np.round(total_acc, 2), fmt='%.3f', padding = 3)
rect2 = ax.bar(br2, male_acc, width = barWidth, label = 'Male Sensitivity', linewidth = 2, color = '#E68B81')
ax.bar_label(rect2, np.round(male_acc, 2), fmt='%.3f', padding = 3)
rect3 = ax.bar(br3, female_acc, width = barWidth, label = 'Female Sensitivity', linewidth = 2, color = '#B7B2D0')
ax.bar_label(rect3, np.round(female_acc, 2), fmt='%.3f', padding = 3)

plt.xticks([r + barWidth for r in range(len(total_acc))], 
        models)

plt.legend(loc = 'upper center', fontsize="12")

plt.ylim([0.60, 1.02])
plt.yticks([0.60 + i * 0.1 for i in range(5)])
plt.ylabel("Sensitivity")
plt.savefig("figures/accuracy/Fig_sensitivity_with_gender.png", dpi=300)
plt.show()

# total_acc = [0.780, 0.805, 0.805, 0.780, 0.878, 0.866, 0.878, 0.878, 0.902]
# male_acc = [0.760, 0.800, 0.860, 0.800, 0.880, 0.820, 0.860, 0.840, 0.880]
# female_acc = [1.000, 0.875, 0.750, 0.625, 0.875, 0.875, 0.875, 0.875, 1.000]

# # set width of bar 
# fig, ax = plt.subplots(1, 1, figsize=(15, 5))

# # Set position of bar on X axis 
# br1 = np.arange(len(total_acc)) 
# br2 = [x + barWidth for x in br1] 
# br3 = [x + barWidth for x in br2] 

# rect1 = ax.bar(br1, total_acc, width = barWidth, label = 'Total Specificity', linewidth = 2, color = '#CE4444')
# ax.bar_label(rect1, np.round(total_acc, 2), fmt='%.3f', padding = 3)
# rect2 = ax.bar(br2, male_acc, width = barWidth, label = 'Male Specificity', linewidth = 2, color = '#DB7014')
# ax.bar_label(rect2, np.round(male_acc, 2), fmt='%.3f', padding = 3)
# rect3 = ax.bar(br3, female_acc, width = barWidth, label = 'Female Specificity', linewidth = 2, color = '#98A654')
# ax.bar_label(rect3, np.round(female_acc, 2), fmt='%.3f', padding = 3)

# plt.xticks([r + barWidth for r in range(len(total_acc))], 
#         models)

# plt.legend(loc = 'best', fontsize="12")

# plt.ylim([0.60, 1.02])
# plt.yticks([0.60 + i * 0.1 for i in range(5)])
# plt.ylabel("Specificity")
# plt.savefig("figures/accuracy/Fig_specificity_with_gender.png", dpi=300)
# plt.show()

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))
# total_gender = [725, 193]
# total_gender_positive = [458, 50]

# train_gender = [626, 186]
# train_gender_positive = [364, 62]

# test_gender = [144, 40]
# test_gender_positive = [94, 8]

# br1 = np.arange(len(total_gender)) 
# ax1.bar(br1, total_gender, width = barWidth, color = '#885B5B', label = 'All labels')
# ax1.bar(br1, total_gender_positive, width = barWidth, color = '#85A646', label = 'Positive labels')
# ax1.set_title("Entire Dataset")
# ax1.set_xticks(br1, ['Male', 'Female'])
# ax1.set_ylim([0, 750])
# ax1.legend(loc = 'best', fontsize = '12')

# ax2.bar(br1, train_gender, width = barWidth, color = '#885B5B', label = 'All labels')
# ax2.bar(br1, train_gender_positive, width = barWidth, color = '#85A646', label = 'Positive labels')
# ax2.set_title("Augmented Train Set")
# ax2.set_xticks(br1, ['Male', 'Female'])
# ax2.set_ylim([0, 650])
# ax2.legend(loc = 'best', fontsize = '12')

# ax3.bar(br1, test_gender, width = barWidth, color = '#885B5B', label = 'All labels')
# ax3.bar(br1, test_gender_positive, width = barWidth, color = '#85A646', label = 'Positive labels')
# ax3.set_title("Test Set")
# ax3.set_xticks(br1, ['Male', 'Female'])
# ax3.set_ylim([0, 150])
# ax3.legend(loc = 'best', fontsize = '12')

# plt.savefig("figures/dataset/Fig_gender_label.png", dpi=300)
# plt.show()

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15, 5))
# barWidth = 0.35
# total_gender = [80, 211, 374, 222, 31]
# total_gender_positive = [26, 85, 212, 163, 22]

# train_gender = [75, 207, 333, 174, 23]
# train_gender_positive = [21, 74, 166, 128, 17]

# test_gender = [15, 35, 76, 51, 7]
# test_gender_positive = [5, 11, 46, 35, 5]

# br1 = np.arange(len(total_gender)) 
# ax1.bar(br1, total_gender, width = barWidth, color = '#885B5B', label = 'All labels')
# ax1.bar(br1, total_gender_positive, width = barWidth, color = '#85A646', label = 'Positive labels')
# ax1.set_title("Entire Dataset")
# ax1.set_xticks(br1, ['39-', '40-49', '50-59', '60-69', '70+'])
# ax1.set_ylim([0, 450])
# ax1.legend(loc = 'best', fontsize = '12')

# ax2.bar(br1, train_gender, width = barWidth, color = '#885B5B', label = 'All labels')
# ax2.bar(br1, train_gender_positive, width = barWidth, color = '#85A646', label = 'Positive labels')
# ax2.set_title("Augmented Train Set")
# ax2.set_xticks(br1, ['39-', '40-49', '50-59', '60-69', '70+'])
# ax2.set_ylim([0, 400])
# ax2.legend(loc = 'best', fontsize = '12')

# ax3.bar(br1, test_gender, width = barWidth, color = '#885B5B', label = 'All labels')
# ax3.bar(br1, test_gender_positive, width = barWidth, color = '#85A646', label = 'Positive labels')
# ax3.set_title("Test Set")
# ax3.set_xticks(br1, ['39-', '40-49', '50-59', '60-69', '70+'])
# ax3.set_ylim([0, 100])
# ax3.legend(loc = 'best', fontsize = '12')

# plt.savefig("figures/dataset/Fig_age_label.png", dpi=300)
# plt.show()

fig, ax = plt.subplots(1, 1, figsize=(20, 5))
barWidth = 0.15

total_acc = [85.33, 87.50, 85.87, 85.87, 88.59, 87.50, 88.04, 88.59, 85.87]
g1_acc = [93.33, 86.67, 100.00, 100.00, 93.33, 80.00, 86.67, 80.00, 86.67]
g2_acc = [82.86, 85.71, 85.71, 88.57, 85.71, 91.43, 91.43, 91.43, 80.00]
g3_acc = [84.21, 88.16, 84.21, 85.53, 90.79, 88.53, 88.16, 86.84, 85.53]
g4_acc = [86.27, 90.20, 86.27, 82.35, 88.24, 92.16, 88.24, 94.12, 90.20]
g5_acc = [85.71, 71.43, 71.43, 71.43, 71.43, 71.43, 71.43, 71.43, 85.71]

# Set position of bar on X axis 
br1 = np.arange(len(total_acc)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
br4 = [x + barWidth for x in br3] 
br5 = [x + barWidth for x in br4] 
br6 = [x + barWidth for x in br5] 

#rect1 = ax.bar(br1, total_acc, width = barWidth, label = 'Total Accuracy', linewidth = 2, color = '#CE4444')
# ax.bar_label(rect1, np.round(total_acc, 1), fmt='%.2f', padding = 3)
rect2 = ax.bar(br2, g1_acc, width = barWidth, label = 'Accuracy [39-]', linewidth = 2, color = '#EAAA60')
# ax.bar_label(rect2, np.round(g1_acc, 1), fmt='%.2f', padding = 3)
rect3 = ax.bar(br3, g2_acc, width = barWidth, label = 'Accuracy [40-49]', linewidth = 2, color = '#E68B81')
# ax.bar_label(rect3, np.round(g2_acc, 1), fmt='%.2f', padding = 3)

rect4 = ax.bar(br4, g3_acc, width = barWidth, label = 'Accuracy [50-59]', linewidth = 2, color = '#B7B2D0')
# ax.bar_label(rect1, np.round(g3_acc, 1), fmt='%.2f', padding = 3)
rect5 = ax.bar(br5, g4_acc, width = barWidth, label = 'Accuracy [60-69]', linewidth = 2, color = '#7DA6C6')
# ax.bar_label(rect2, np.round(g4_acc, 1), fmt='%.2f', padding = 3)
rect6 = ax.bar(br6, g5_acc, width = barWidth, label = 'Accuracy [70+]', linewidth = 2, color = '#84C3B7')
# ax.bar_label(rect3, np.round(g5_acc, 1), fmt='%.2f', padding = 3)

ax.axhline(y = 87.02, linestyle = '--', lw = 1.5)

plt.xticks([r + 3*barWidth for r in range(len(total_acc))], 
        models)

plt.legend(loc = 'best', fontsize="10")

plt.ylim([70, 102])
plt.yticks([70 + i * 5 for i in range(7)])
plt.ylabel("Accuracy [%]")
plt.savefig("figures/accuracy/Fig_acc_with_age.png", dpi=300)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(20, 5))
barWidth = 0.15

total_acc = [85.33, 87.50, 85.87, 85.87, 88.59, 87.50, 88.04, 88.59, 85.87]
g1_acc = [0.800, 0.800, 1.000, 1.000, 0.800, 0.600, 0.600, 0.600, 0.600]
g2_acc = [0.818, 0.818, 0.818, 0.909, 0.636, 0.818, 0.818, 0.818, 0.455]
g3_acc = [0.891, 0.935, 0.870, 0.913, 0.913, 0.870, 0.870, 0.870, 0.826]
g4_acc = [0.971, 1.000, 0.971, 0.943, 0.971, 0.971, 0.971, 1.000, 0.943]
g5_acc = [1.000, 0.800, 0.800, 0.800, 0.800, 0.800, 0.800, 0.800, 1.000]

# Set position of bar on X axis 
br1 = np.arange(len(total_acc)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
br4 = [x + barWidth for x in br3] 
br5 = [x + barWidth for x in br4] 
br6 = [x + barWidth for x in br5] 

#rect1 = ax.bar(br1, total_acc, width = barWidth, label = 'Total Accuracy', linewidth = 2, color = '#CE4444')
# ax.bar_label(rect1, np.round(total_acc, 1), fmt='%.2f', padding = 3)
rect2 = ax.bar(br2, g1_acc, width = barWidth, label = 'Sensitivity [39-]', linewidth = 2, color = '#EAAA60')
# ax.bar_label(rect2, np.round(g1_acc, 1), fmt='%.2f', padding = 3)
rect3 = ax.bar(br3, g2_acc, width = barWidth, label = 'Sensitivity [40-49]', linewidth = 2, color = '#E68B81')
# ax.bar_label(rect3, np.round(g2_acc, 1), fmt='%.2f', padding = 3)

rect4 = ax.bar(br4, g3_acc, width = barWidth, label = 'Sensitivity [50-59]', linewidth = 2, color = '#B7B2D0')
# ax.bar_label(rect1, np.round(g3_acc, 1), fmt='%.2f', padding = 3)
rect5 = ax.bar(br5, g4_acc, width = barWidth, label = 'Sensitivity [60-69]', linewidth = 2, color = '#7DA6C6')
# ax.bar_label(rect2, np.round(g4_acc, 1), fmt='%.2f', padding = 3)
rect6 = ax.bar(br6, g5_acc, width = barWidth, label = 'Sensitivity [70+]', linewidth = 2, color = '#84C3B7')
# ax.bar_label(rect3, np.round(g5_acc, 1), fmt='%.2f', padding = 3)

ax.axhline(y = 0.893, linestyle = '--', lw = 1.5)

plt.xticks([r + 3*barWidth for r in range(len(total_acc))], 
        models)

plt.legend(loc = 'best', fontsize="10")

plt.ylim([0.40, 1.02])
plt.yticks([0.40 + i * 0.1 for i in range(7)])
plt.ylabel("Sensitivity")
plt.savefig("figures/accuracy/Fig_sensitivity_with_age.png", dpi=300)
plt.show()