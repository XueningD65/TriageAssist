import pandas as pd

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category = FutureWarning)

import matplotlib.pyplot as plt



df = pd.read_csv('cleveland.csv', header = None)

df.columns = ['age', 'sex', 'Chest Pain', 'rest bps', 'cholesterol',
              'fbs', 'restecg', 'thalach', 'exang', 
              'oldpeak', 'slope', 'ca', 'thal', 'target']

print(df['sex'])
df['sex'].value_counts().plot(kind='bar')
plt.xticks([0, 1], ['Male', 'Female'])
plt.xticks(rotation=0)
plt.xlabel("Sex")
plt.title("Distribution of Sex Groups across the dataset")
plt.show()

import seaborn as sns
corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.xticks(rotation=40)
plt.show()

# sns.pairplot(data=df,hue='sex')
# plt.show()