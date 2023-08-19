#!/usr/bin/env python
# coding: utf-8

# # AMOD KUMAR
# ## Task 5
# ### House Price Prediction

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


# In[8]:


df = pd.read_csv('BHP.csv')


# In[10]:


df.head()


# In[9]:


df.shape


# In[14]:


df.info()


# In[26]:


df.columns


# ## DATA CLEANING

# In[18]:


df.isnull().sum()


# In[52]:


df=df.drop(['area_type','availability','balcony','society'],axis=1)
df


# In[54]:


df=df.dropna()


# In[55]:


df.isnull().sum()


# In[56]:


df.shape


# In[63]:


df['size'].unique()


# # Feature Engineering
# ## Add new feature(integer) for bhk (Bedrooms Hall Kitchen)

# In[65]:


df['BHK']=df['size'].apply(lambda x: int(x.split(' ')[0]))


# In[67]:


df.head()


# In[68]:


df['BHK'].unique()


# In[69]:


df[df.BHK>20]


# In[70]:


df.total_sqft.unique()


# In[71]:


def isfloat(x):
    try:
        float(x)
    except:
        return False
    return True


# In[72]:


df[~df['total_sqft'].apply(isfloat)].head(10)


# In[73]:


def convert_sqft_tonum(x):
    token=x.split('-')
    if len(token)==2:
        return (float(token[0])+float(token[1]))/2
    try:
        return float(x)
    except:
        return None


# In[74]:


df=df.copy()
df['total_sqft']=df['total_sqft'].apply(convert_sqft_tonum)


# In[75]:


df.head(10)


# In[78]:


df.loc[672]


# ## Add new feature called price per square feet

# In[80]:


df1=df.copy()
df1['price_per_sqft']=df1['price']*1000000/df1['total_sqft']
df1.head()


# In[81]:


len(df1.location.unique())


# In[83]:


df1.location = df1.location.apply(lambda x : x.strip())


# In[86]:


unique_location_counts = df1.groupby('location')['location'].agg('count').sort_values(ascending=False)


# In[87]:


unique_location_counts


# In[88]:


len(df1.location.unique())


# In[93]:


len(unique_location_counts[unique_location_counts<=10])


# In[94]:


locationlessthan10=unique_location_counts[unique_location_counts<=10]
locationlessthan10


# In[95]:


len(df1.location.unique())


# In[97]:


df1.location=df1.location.apply(lambda x: 'other' if x in locationlessthan10 else x)
len(df1.location.unique())


# In[98]:


df1.head(10)


# In[99]:


df1[df1.total_sqft/df1.BHK<300].head()


# In[101]:


df2=df1[~(df1.total_sqft/df1.BHK<300)]
df2.head(10)


# In[102]:


df2.shape


# ## Outlier Removal Using Standard Deviation and Mean

# In[103]:


df2["price_per_sqft"].describe().apply(lambda x:format(x,'f'))


# In[107]:


def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st))& (subdf.price_per_sqft<(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df3=remove_pps_outliers(df2)
df3.shape


# In[108]:


import matplotlib.pyplot as plt
def plot_scatter_chart(df,location):
    bhk2=df[(df.location==location)&(df.BHK==2)]
    bhk3=df[(df.location==location)&(df.BHK==3)]
    plt.rcParams['figure.figsize']=(15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='Blue',label='2 BHK',s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,color='green',marker='+',label='3 BHK',s=50)
    plt.xlabel('Total Square Foot')
    plt.ylabel('Price')
    plt.title(location)
    plt.legend()
plot_scatter_chart(df3,"Rajaji Nagar")


# In[109]:


def remove_bhk_outliers(df):
    exclude_indices=np.array([])
    for location, location_df in df.groupby('location'):
        bhk_sats={}
        for BHK,BHK_df in location_df.groupby('BHK'):
            bhk_sats[BHK]={
                'mean':np.mean(BHK_df.price_per_sqft),
                'std':np.std(BHK_df.price_per_sqft),
                'count':BHK_df.shape[0]
            }
        for BHK,BHK_df in location_df.groupby('BHK'):
            stats=bhk_sats.get(BHK-1)
            if stats and stats['count']>5:
                exclude_indices=np.append(exclude_indices,BHK_df[BHK_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')

df4=remove_bhk_outliers(df3)
df4.shape


# In[111]:


plot_scatter_chart(df4,"Rajaji Nagar")


# In[114]:


plt.rcParams['figure.figsize']=(10,6)
plt.hist(df4.price_per_sqft,rwidth=0.6)
plt.xlabel("Price Per Square Foor")
plt.ylabel("Count")


# # Outlier Removal Using Bathrooms Feature

# In[115]:


df4.bath.unique()


# In[117]:


df4[df4.bath>10]


# In[119]:


plt.rcParams['figure.figsize']=(10,6)
plt.hist(df4.bath,rwidth=0.6)
plt.xlabel("Number Of Bathroom")
plt.ylabel("Count")


# In[120]:


df4[df4.bath>df4.BHK+2]


# In[121]:


df5=df4[df4.bath<df4.BHK+2]
df5.shape


# In[122]:


df6=df5.drop(['size','price_per_sqft'],axis='columns')
df6


# In[123]:


dummies=pd.get_dummies(df6.location)
dummies.head(10)


# In[124]:


df7=pd.concat([df6,dummies.drop('other',axis='columns')],axis='columns')
df7.head()


# In[127]:


df8=df7.drop('location',axis='columns')
df8.head()


# ## Model building

# In[128]:


df8.shape


# In[129]:


X=df8.drop('price',axis='columns')
X.head()


# In[130]:


y=df8.price


# In[131]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[132]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)


# In[133]:


# Use K Fold cross validation to measure accuracy of our LinearRegression model
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)


# In[136]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit
import pandas as pd

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {}
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

# Assuming you have X and y defined
results = find_best_model_using_gridsearchcv(X, y)
print(results)


# In[137]:


#Based on above results we can say that LinearRegression gives the best score. Hence we will use that.


# In[138]:


def price_predict(location,sqft,bath,BHK):
    loc_index=np.where(X.columns==location)[0][0]
    x=np.zeros(len(X.columns))
    x[0]=sqft
    x[1]=bath
    x[2]=BHK
    if loc_index >=0:
        x[loc_index]=1
    return model.predict([x])[0]


# In[139]:


price_predict('1st Phase JP Nagar',1000,2,2)


# In[140]:


price_predict('1st Phase JP Nagar',1000,2,3)


# In[141]:


price_predict('5th Phase JP Nagar',1000,2,2)


# In[142]:


price_predict('Indira Nagar',1000,2,2)


# In[ ]:




