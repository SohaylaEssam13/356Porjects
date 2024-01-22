#!/usr/bin/env python
# coding: utf-8

# # Simple linear regression

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set() #to override all matplotlibs


# In[13]:


data = pd.read_csv('1.01. Simple linear regression.csv')


# In[14]:


data


# In[15]:


data.describe()


# In[16]:


y = data['GPA']
x1 = data['SAT']


# In[17]:


plt.scatter(x1,y)
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize =20)
plt.show()


# In[18]:


x0 = sm.add_constant(x1)


# In[19]:


results = sm.OLS(y,x0).fit()


# In[20]:


results.summary()


# In[27]:


# Create a scatter plot
plt.scatter(x1,y)
# Define the regression equation, so we can plot it later
yhat = 0.0017*x1 + 0.275
# Plot the regression line against the independent variable (SAT)
fig = plt.plot(x1,yhat, lw=4, c='orange', label ='regression line')
# Label the axes
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()

