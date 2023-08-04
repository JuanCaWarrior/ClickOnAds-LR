#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


ad_data = pd.read_csv('advertising.csv')
ad_data.head()


# In[3]:


ad_data.describe()


# In[4]:


ad_data.info()


# In[5]:


sns.pairplot(ad_data, hue= 'Clicked on Ad', palette='bwr')


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[9]:


from sklearn.linear_model import LogisticRegression


# In[10]:


logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# In[11]:


predictions =logmodel.predict(X_test)


# In[12]:


from sklearn.metrics import classification_report, confusion_matrix


# In[13]:


print(classification_report(y_test, predictions))
print(confusion_matrix(y_test,predictions))


# In[ ]:




