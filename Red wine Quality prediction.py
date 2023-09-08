#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('winequality-red'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


#Loading dataset
wine = pd.read_csv(r'C:\Users\hverm\Downloads/winequality-red.csv')


# In[3]:


#lets check the dataset
wine.head()


# In[4]:


print(wine.info())
#no missing values


# In[5]:


#EDA
# # # # #univarite analysis***


# In[6]:


catgorical_feat=[features for features in wine.columns if wine[features].nunique()<10]
catgorical_feat


# In[7]:


print(wine.quality.value_counts())
print(wine.quality.value_counts(normalize=True))
sns.countplot(x='quality',data=wine)


# In[8]:


wine.hist(figsize = (10,10),color="b",bins=40,alpha=1)


# In[9]:


# # # # we have some skewed features


# In[10]:


wine.describe()


# In[11]:


wine.columns


# In[13]:


cont_feat=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol', 'quality']


# In[14]:


wine[cont_feat].plot.box(figsize=(20,10))

plt.show()
#there are also some outliers also


# In[15]:


# Numerical vs Target Variable


# In[16]:


sns.pairplot(wine,hue='quality')


# In[17]:


plt.figure(figsize = (18,12))
sns.heatmap(wine.corr(), annot = True, cmap = "RdYlGn")

plt.show()


# In[18]:


#Here we see that fixed acidity does not give any specification to classify the quality.
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)


# In[19]:


#Here we see that its quite a downing trend in the volatile acidity as we go higher the quality 
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)


# In[20]:


#Composition of citric acid go higher as we go higher in the quality of the wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = wine)


# In[21]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = wine)


# In[22]:


#Composition of chloride also go down as we go higher in the quality of the wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = wine)


# In[23]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine)


# In[24]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = wine)


# In[25]:


#Sulphates level goes higher with the quality of wine
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = wine)


# In[26]:


#Alcohol level also goes higher as te quality of wine increases
fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = wine)


# In[27]:


# data preprocessing


# In[28]:


#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)


# In[29]:


from sklearn.preprocessing import StandardScaler, LabelEncoder
#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()


# In[30]:


#Bad becomes 0 and good becomes 1 
wine['quality'] = label_quality.fit_transform(wine['quality'])


# In[31]:


wine['quality'].value_counts()


# In[32]:


sns.countplot(wine['quality'])


# In[33]:


#Now seperate the dataset as response variable and feature variabes
X = wine.drop('quality', axis = 1)
y = wine['quality']


# In[34]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')
#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[35]:


#Applying Standard scaling to get optimized result
sc = StandardScaler()


# In[36]:


X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[37]:


# Model Selection
# Logistic Regression


# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
clf = LogisticRegression()
scoring = 'accuracy'
score = cross_val_score(clf, X_train, y_train, n_jobs=1, scoring=scoring)
print(score)


# In[39]:


round(np.mean(score)*100, 2)


# In[40]:


from sklearn.model_selection import KFold
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# In[41]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 15)
scoring = 'accuracy'
score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[42]:


# kNN Score
round(np.mean(score)*100, 2)


# In[43]:


# Decision Tree


# In[44]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[45]:


# decision tree Score
round(np.mean(score)*100, 2)


# In[46]:


# Random Forest


# In[47]:


from sklearn.ensemble import RandomForestClassifier
rnd = RandomForestClassifier(n_estimators=45)
scoring = 'accuracy'
score = cross_val_score(rnd, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[48]:


round(np.mean(score)*100, 2)


# In[49]:


# Naive Bayes


# In[50]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[51]:


round(np.mean(score)*100, 2)


# In[52]:


# Support Vector Classifier


# In[53]:


from sklearn.svm import SVC
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[54]:


round(np.mean(score)*100, 2)


# In[55]:


# **Let's try to increase our accuracy of models
#   Grid Search CV**


# In[56]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint


# In[57]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)
{'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
{'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}


# In[59]:


## testing


# In[60]:


rnd=RandomForestClassifier(n_estimators=50)
rnd.fit(X_train,y_train)
#test_data = test.drop( "Loan_ID", axis=1).copy()
prediction = rnd.predict(X_test)


# In[61]:


print(classification_report(y_test, prediction))


# In[62]:


rfc_eval = cross_val_score(estimator = rnd, X = X_train, y = y_train, cv = 6)
rfc_eval.mean()


# In[64]:


print("Thanku")

