"""
Created on Thu May  7 15:23:13 2020
@author: DESHMUKH
RANDOM FOREST
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
pd.set_option('display.max_columns',None)

# =================================================================================
# Business Problem - Create a Random Forest Model to classify 'Class Variable'.
# =================================================================================

diabetes = pd.read_csv('Diabetes.csv',skipinitialspace = True)
diabetes.head()
diabetes.isnull().sum()
diabetes.columns = diabetes.columns.str.replace(' ', '_')
diabetes.head()

# Summary
diabetes.describe()

# Boxplot
diabetes.boxplot(notch=True, patch_artist=True, grid=False);plt.xticks(fontsize=4,rotation = 30)

# Histrogram
diabetes.hist(grid=False)

# Checking Percentage of Output classes with the help Value Count.
(diabetes['Class_variable'].value_counts())/len(diabetes)*100  # No-65%,Yes-35%

################################ - Spliting data in X and y - ################################

X = diabetes.iloc[:,:8]
y = diabetes.iloc[:,8]

############################# - Spliting data in train and test - ############################

# Stratified random sampling becuase output have inbalanced Data
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.30, random_state = False, stratify = y )

# Rechecking Percentage of Output classes by using values counts. 
(y_train.value_counts()/len(y_train))*100 # No-65%,Yes-35% Percentage of sample Output classes simillier to population 

################################## - Random Forest Model - ###################################

rf = RandomForestClassifier(n_estimators=35, criterion='entropy')
rf.fit(X_train,y_train)

# Prediction on Train & Test Data
pred_train = rf.predict(X_train)
pred_test = rf.predict(X_test)

# Accuracy of Train and Test
accuracy_score(y_train,pred_train) 
accuracy_score(y_test,pred_test) 

# Confusion matrix of Train and Test
## Train
confusion_matrix_train = pd.crosstab(y_train,pred_train,rownames=['Actual'],colnames= ['Predictions']) 
sns.heatmap(confusion_matrix_train, annot = True, cmap = 'YlGn',fmt='g')

## Test
confusion_matrix_test = pd.crosstab(y_test,pred_test,rownames=['Actual'],colnames= ['Predictions']) 
sns.heatmap(confusion_matrix_test, annot = True, cmap = 'PuRd',fmt='g')

################################# - Visualizing Decision Trees - ###################################

#from sklearn.tree import plot_tree
plt.figure(figsize=(25,10))
a = plot_tree(rf.estimators_[1],                # You can change estimators_ value to plot different decision tree out of 35 
              feature_names = X_train.columns, 
              class_names = diabetes.columns[2], 
              filled=True, 
              rounded=True,#fontsize=14
              )

                         # ---------------------------------------------------- #



