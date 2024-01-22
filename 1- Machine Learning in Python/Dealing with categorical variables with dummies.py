#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.api as sm
import pandas as pd


# In[2]:


raw_data = pd.read_csv('1.03. Dummies.csv')
raw_data


# In[3]:


data = raw_data.copy()
data['Attendance'] = data['Attendance'].map({'Yes':1, 'No':0})
data


# In[4]:


x1 = data[['SAT', 'Attendance']]
y = data['GPA']


# In[5]:


x0 = sm.add_constant(x1)
results = sm.OLS(y,x0).fit()


# In[6]:


results.summary()


# In[7]:


plt.scatter(data['SAT'], y)
yhat_yes = 0.8665 + 0.0014*data['SAT']
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat = 0.2750 + (0.0017*data['SAT'])
fig = plt.plot(data['SAT'], yhat_yes, lw = 2, c= 'Red')
fig = plt.plot(data['SAT'], yhat_no, lw = 2, c='Green')
fig = plt.plot(data['SAT'], yhat, lw =2 , c= 'Blue')   
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()


# In[8]:


new_data = pd.DataFrame({'const':1, 'SAT':[1700,1760], 'Attendance':[0,1]})
new_data 


# In[9]:


new_data.rename({0:'Bob', 1:'Alice'})


# In[10]:


new_data


# In[11]:


predictions = results.predict(new_data)
predictions


# In[12]:


predictionsdf = pd.DataFrame({'Predictions':predictions})
joined = new_data.join(predictionsdf )
joined.rename({0:'Bob', 1:'Alice'})


# In[ ]:




