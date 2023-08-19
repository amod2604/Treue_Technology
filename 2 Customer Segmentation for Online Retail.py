#!/usr/bin/env python
# coding: utf-8

# ## Amod Kumar 
# #  Task 2
# #  Customer Segmentation for Online Retail
# 

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


ds = pd.read_excel("Online Retail.xlsx")


# In[5]:


ds.keys() 


# In[6]:


ds.head()


# ## EDA

# In[7]:


#data cleaning
print(ds.info())


# In[8]:


ds.describe()


# In[9]:


ds=ds.loc[ds["Quantity"] >0] 


# In[10]:


ds.shape 


# In[11]:


ds.describe()


# ## Recency,Frequency,Monetary

# In[12]:


#calculating our monetary value

ds["Sale"] =ds.Quantity * ds.UnitPrice 
ds.head()
monetary =ds.groupby("CustomerID").Sale.sum()
monetary = monetary.reset_index() 


# In[13]:


monetary.head()


# In[14]:


#calculating our frequency

frequency=ds.groupby("CustomerID").InvoiceNo.count()
frequency = frequency.reset_index()
frequency


# In[15]:


#calculating our recency value

LastDate=max(ds.InvoiceDate) 
LastDate
LastDate = LastDate + pd.DateOffset(days=1)
LastDate
ds["Diff"] = LastDate - ds.InvoiceDate
recency = ds.groupby("CustomerID").Diff.min()
recency = recency.reset_index()


# In[16]:


recency.head()


# In[17]:


#comnining all dataframes into one
rmf = monetary.merge(frequency, on = "CustomerID")
rmf = rmf.merge(recency, on = "CustomerID")
rmf.columns = ["CustomerID", "Monetary", "Frequence", "Recency"]
rmf
RMF1 = rmf.drop("CustomerID",axis =1) 
RMF1.Recency = RMF1.Recency.dt.days


# In[18]:


RMF1


# In[19]:


from sklearn.cluster import KMeans


# In[34]:


import warnings
from sklearn.cluster import KMeans

def ignore_future_warning(message, category, filename, lineno, file=None, line=None):
    if "The default value of `n_init` will change from 10 to 'auto' in 1.4." in str(message):
        return True
    return False

warnings.filterwarnings("ignore", category=FutureWarning)  # Only specify the category

kmeans = KMeans(n_clusters=3)
ssd = []

for k in range(1, 20):
    km = KMeans(n_clusters=k)
    km.fit(RMF1)
    ssd.append(km.inertia_)


# In[35]:


plt.plot(np.arange(1,20), ssd,color="darkblue")
plt.scatter(np.arange(1,20), ssd,color="red")
plt.show()


# In[22]:


model = KMeans(n_clusters=5)


# In[36]:


ClusterID = model.fit_predict(RMF1)


# In[37]:


ClusterID


# In[38]:


RMF1["ClusterID"] = ClusterID


# In[39]:


RMF1


# In[40]:


km_cluster_sale =RMF1.groupby("ClusterID").Monetary.mean()
km_cluster_Recency =RMF1.groupby("ClusterID").Recency.mean()
km_cluster_Frequence =RMF1.groupby("ClusterID").Frequence.mean()
km_cluster_sale


# In[41]:


import seaborn as sns

#first we are plotting bar chart 
fig, axs = plt.subplots(1,3, figsize = (15, 5))
sns.barplot(x = [0,1,2,3,4],  y = km_cluster_sale , ax = axs[0])
sns.barplot(x = [0,1,2,3,4],  y = km_cluster_Frequence , ax = axs[1])
sns.barplot(x = [0,1,2,3,4],  y = km_cluster_Recency , ax = axs[2])


# In[42]:


fig,axis = plt.subplots(1,3, figsize =(18,5))
ax1 =fig.add_subplot(1,3,1)
plt.title("Monetary Mean")
ax1.pie(km_cluster_sale, labels =[0,1,2,3,4])
ax1 =fig.add_subplot(1,3,2)
plt.title("Frequency Mean")
ax1.pie(km_cluster_Frequence, labels =[0,1,2,3,4])
ax1 =fig.add_subplot(1,3,3)
plt.title("Recency Mean")
ax1.pie(km_cluster_Recency, labels =[0,1,2,3,4])
plt.show()


# # from the above pie chart we can easily understand our 5 groups according to Recency mean,Frequency mean and Monetary mean.
# # Group 1 is the group of customer who spends maximum amount of money and also has a good frequency and low recency rate.Group 4 are the customers whose frequency rate is maximum and monetary value is also good and recency rate is also quite good, whereas Group 0 is the group of customers who has a very high recency rate means they have not purchased anything from the past.

# In[ ]:




