#!/usr/bin/env python
# coding: utf-8

# # Practical Example

# ## Importing  relevant libraries 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
import pandas as pd


# In[2]:


raw_data = pd.read_csv('1.04. Real-life example.csv')


# ## Preprocessing

# ### Raw Data - Exploring the describtive statistics of the variables-

# In[3]:


raw_data.describe(include = 'all')


# ### Determining the variable of interest

# In[4]:


data = raw_data.drop(['Model'], axis = 1)
data.describe(include ='all')


# ### Dealing with missing value

# In[5]:


data.isnull()


# In[6]:


data.isnull().sum()


# In[7]:


data_no_mv = data.dropna(axis = 0)


# In[8]:


data_no_mv


# In[9]:


data_no_mv.describe(include = 'all')


# ## Exploring the PDFs

# In[10]:


sns.displot(data_no_mv['Price'])


# ### Dealing with outliers

# In[11]:


q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price']<q]
data_1.describe(include = 'all')


# In[12]:


sns.displot(data_1['Price'])


# In[13]:


sns.displot(data_no_mv['Mileage'])


# In[14]:


q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_no_mv['Mileage']<q]
data_2.describe(include = 'all')


# In[15]:


sns.displot(data_2['Mileage'])


# In[16]:


sns.displot(data_no_mv['EngineV'])


# In[17]:


EngV = pd.DataFrame(raw_data['EngineV'])
EngV = EngV.dropna(axis =0)


# In[18]:


EngV.sort_values(by ='EngineV')


# In[19]:


data_3 = data_2[data_2['EngineV']<6.5]


# In[20]:


sns.displot(data_3['EngineV'])


# In[21]:


sns.displot(data_no_mv['Year'])


# In[22]:


q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]


# In[23]:


sns.displot(data_4['Year'])


# In[24]:


data_cleaned = data_4.reset_index(drop = True)


# In[25]:


data_cleaned.describe(include = 'all')


# In[26]:


f,(ax1,ax2,ax3) = plt.subplots(1,3,  sharey = True,figsize= (15,3))
#f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax2.set_title('Price and Engine')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax3.set_title('Price and Mileage')
plt.show()


# In[27]:


sns.displot(data_cleaned['Price'])


# ## Relaxing Assumption

# ###### Make the price linear by using log

# In[28]:


log_price = np.log(data_cleaned['Price'])
data_cleaned['Log_price'] = log_price
data_cleaned


# In[29]:


f,(ax1,ax2,ax3) = plt.subplots(1,3,  sharey = True,figsize= (15,3))
#f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'], data_cleaned['Log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['Log_price'])
ax2.set_title('Log Price and Engine')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['Log_price'])
ax3.set_title('Log Price and Mileage')
plt.show()


# In[30]:


data_cleaned = data_cleaned.drop(['Price'], axis = 1)


# ## Multicolinearity

# #### We need variance inflation factor from statsmodel

# In[31]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
#VIF compare between at least three variables 


# In[32]:


variables = data_cleaned[['Mileage', 'Year', 'EngineV']]


# In[33]:


vif = pd.DataFrame()


# In[34]:


vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]


# In[35]:


vif["Features"]= variables.columns


# In[36]:


vif


# In[37]:


data_no_multicollinearity = data_cleaned.drop(['Year'], axis =1)


# ## Created dummy variables

# #### Pandas gives us a library  to deal with dummies 

# In[38]:


# we must drop the first dummy while if the first dummy 
data_with_dummies =pd.get_dummies(data_no_multicollinearity, drop_first = True)


# In[39]:


data_with_dummies.head()


# ### Rearrange bit

# In[40]:


data_with_dummies.columns.values


# In[41]:


cols = [
    'Log_price','Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes'
    
]


# In[42]:


data_preprocessed = data_with_dummies[cols]


# In[43]:


data_preprocessed.head()


# ## Linear Regression Model 

# ###### Standarize the data and make regression

# ### Declare the inputs and targets

# In[44]:


targets = data_preprocessed['Log_price']
inputs = data_preprocessed.drop(['Log_price'], axis = 1)


# ### Scale the data

# In[45]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(inputs)


# In[46]:


input_scaled = scaler.transform(inputs)


# # Training and testing

# ## Train Test Split

# In[47]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(input_scaled, targets, test_size= 0.2, random_state = 365)


# # Create the regression

# In[48]:


reg = LinearRegression()
reg.fit(x_train, y_train)


# In[49]:


## our predicted value, It is not linear it is log linear
# to check the result plot the predicted value of the regression against observed value
y_hat = reg.predict(x_train)


# In[50]:


plt.scatter(y_train, y_hat)
plt.xlabel('Targest (y_train)', size = 18)
plt.ylabel('Predictions (y_hat)', size = 18)
plt.ylim(6,13)
plt.xlim(6,13)
plt.show()
# here we want if the target is 7 the predicted must be 7, as figured out here the data is ot perfect but it is not random


# In[51]:


#another popular check is the residuals plo, the residuals are the difference bteween the targts and the predictions
sns.displot(y_train - y_hat)
plt.title("Residuals PDF", size = 18)
# There is much longer tail on the negative side
#which means there are certain observations that (y-train - y-hat) is much lower than the mean which means those tend to overestimate the targets 
#much higher price is predicted than is observed
#accourdning to on the right hand-side is less which means it is rearly underestiamtes the target


# In[52]:


#Calculate the R-squared
reg.score(x_train, y_train)
#our model is explaining 75% of the vriability of the data


# # Find the weights and bias

# In[53]:


reg.intercept_


# In[54]:


reg.coef_


# ### Summary table

# In[55]:


reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights']= reg.coef_
reg_summary

#some weights would be positive other would be negative
#all variables are standarized including the dummies
#postive values means the higher it the higher the price


# In[56]:


#For dummy variable is different
data_cleaned['Brand'].unique()
#when we look to the output variable we would relize that Audi is the dropped one
#So, Audi is the benchmark
#if there is a dummy variable which is positive that means it is more expensive than Audi


# # Testing 

# In[57]:


#Start our test-part by finding predictions
y_hat_test = reg.predict(x_test)


# In[58]:


plt.scatter(y_test, y_hat_test, alpha = 0.2) #alpha the more staurated of colors the higher concentration of point
plt.xlabel('Targest (y_test)', size = 18)
plt.ylabel('Predictions (y_ha_test)', size = 18)
plt.ylim(6,13)
plt.xlim(6,13)
plt.show()


# In[59]:


#making a variable that stands for perfromance
df_pf = pd.DataFrame(y_hat_test, columns=['Prediction'])
df_pf.head()
#this a preditction for a log price 
#since the log is the opposite of the exponential then


# In[60]:


df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
df_pf.head()


# In[61]:


# we want to place the target near to the prediction, to compare them
df_pf['Target'] = np.exp(y_test)
df_pf.head()
# we will have a lot of missing values, which is randomly spread


# In[62]:


# to understand this display the y_test dataFRame
y_test
# we will find it contains indexes, when we split our data into train set and test set the original indeces where preserved
# pandas wants to match indices
# we want to frogot the original indxes


# In[63]:


y_test = y_test.reset_index(drop = True)
y_test


# In[64]:


df_pf['Target'] = np.exp(y_test)
df_pf.head()


# In[65]:


# create Residual which would be the difference between target and prediction
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
#minimizing min(SSE)


# In[66]:


df_pf['Difference%'] = np.absolute(df_pf['Residual']/ df_pf['Target']*100)
df_pf


# In[67]:


df_pf.describe()


# In[68]:


pd.options.display.max_rows = 999
pd.set_option('display.float_format', lambda x: '%.2f' %x)
df_pf.sort_values(by = ['Difference%'])

# the meaning of the resiudal is negative it is the e


# In[ ]:




