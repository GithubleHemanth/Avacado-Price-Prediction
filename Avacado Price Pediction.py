#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\HOME\Desktop\avocado.csv")


# In[3]:


df.head()


# In[4]:


df.drop("Unnamed: 0",axis=1,inplace=True)


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df["Date"]=pd.to_datetime(df["Date"])


# In[10]:


df.info()


# In[11]:


df["type"].value_counts()


# In[12]:


sns.countplot(data=df,x="type")


# In[13]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)


# In[14]:


df.columns


# In[15]:


sns.histplot(df["AveragePrice"],kde=True)


# In[16]:


plt.figure(figsize=(10,8))
sns.lineplot(data=df,x="year",y="Total Volume")


# In[17]:


plt.figure(figsize=(10,8))
sns.lineplot(data=df,x="year",y="AveragePrice")


# In[18]:


df


# In[19]:


df


# In[20]:


plt.figure(figsize=(8,8))
sns.heatmap(df.corr(),cmap="coolwarm",annot=True)


# In[26]:


greater_corr=[]
treshold=0.9
for col1 in df.corr().columns:
    for col2 in df.corr().columns:
        if (abs(df.corr().loc[col1,col2])>treshold) and (col1!=col2):
            greater_corr.append((col1,col2))


# In[27]:


greater_corr


# In[28]:


#4046,4225,small bags,largebags


# In[30]:


final=df.drop(["4046","4225","Small Bags","Large Bags"],axis=1)


# In[31]:


final.head()


# In[33]:


final["type"].value_counts()


# In[34]:


final["type"]=final["type"].map({"conventional":0,"organic":1})


# In[36]:


final.region.value_counts()


# In[37]:


final.drop("region",axis=1,inplace=True)


# In[38]:


final.head()


# In[39]:


final.year.value_counts()


# In[41]:


final_=pd.get_dummies(final,columns=["year"],prefix="year")


# In[43]:


final_.drop("Date",axis=1,inplace=True)


# In[44]:


final_


# In[45]:


from sklearn.preprocessing import StandardScaler


# In[46]:


sc=StandardScaler()


# In[53]:


X=final_.drop("AveragePrice",axis=1)
X


# In[47]:


from sklearn.model_selection import train_test_split


# In[54]:


X=sc.fit_transform(X)


# In[56]:


y=final_.AveragePrice


# In[57]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[58]:


X_train


# In[59]:


from pycaret.regression import *


# In[61]:


exp=setup(data=final_,target="AveragePrice")


# In[62]:


best=compare_models()


# In[63]:


from sklearn.linear_model import LinearRegression


# In[64]:


lr=LinearRegression()


# In[65]:


lr.fit(X_train,y_train)


# In[66]:


y_pred=lr.predict(X_test)


# In[72]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[78]:


mse=mean_squared_error(y_test,y_pred)
mse


# In[80]:


mae=mean_absolute_error(y_test,y_pred)
mae


# In[81]:


r2=r2_score(y_test,y_pred)
r2


# In[84]:


y_test


# In[85]:


y_pred


# In[86]:


df


# In[87]:


from sklearn.decomposition import PCA


# In[88]:


pca=PCA(0.95)


# In[89]:


X


# In[ ]:




