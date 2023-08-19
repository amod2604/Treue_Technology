#!/usr/bin/env python
# coding: utf-8

# # Amod Kumar
# ## Task 4
# ### Email Spam Detection

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('spam.csv', encoding='latin-1')


# In[3]:


df.sample(10)


# In[4]:


df.shape


# # Data Cleaning

# In[5]:


df.info()


# In[6]:


df.drop(columns =['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace = True)


# In[7]:


df.sample(5)


# In[8]:


New_df = df.rename(columns = {'v1':'Target','v2':'Text'})


# In[9]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit_transform(New_df ["Target"])


# In[10]:


New_df ["Target"] = encoder.fit_transform(New_df ["Target"])


# In[11]:


New_df.head()


# In[12]:


#missing value check 
New_df.isnull().sum()


# In[13]:


New_df.duplicated().sum()


# In[14]:


New_df.shape


# In[15]:


New_df = New_df.drop_duplicates(keep ='first')


# In[16]:


New_df.shape


# In[17]:


New_df.duplicated().sum()


# # EDA

# In[18]:


New_df.head()


# In[19]:


New_df['Target'].value_counts()


# In[20]:


import matplotlib.pyplot as plt
plt.pie(New_df['Target'].value_counts(),labels=['ham','spam'],autopct='%0.2f')
plt.show


# In[21]:


import nltk
nltk.download('punkt')


# In[22]:


New_df['Total letters']=New_df['Text'].apply(len) 


# In[23]:


New_df.head()


# In[24]:


New_df['Text'].apply(lambda x:nltk.word_tokenize(x))


# In[25]:


New_df['Total word']=New_df['Text'].apply(lambda x:len(nltk.word_tokenize(x)) ) 


# In[26]:


New_df.head()


# In[27]:


New_df['Total Sentence']=New_df['Text'].apply(lambda x:len(nltk.sent_tokenize(x)) ) 


# In[28]:


New_df.head()


# In[29]:


New_df[['Total letters','Total word','Total Sentence']].describe()


# In[30]:


#Detail about Ham msgs
New_df[New_df['Target']==0][['Total letters','Total word','Total Sentence']].describe()


# In[31]:


#Detail about spam msgs
New_df[New_df['Target']==1][['Total letters','Total word','Total Sentence']].describe()


# In[32]:


import seaborn as sns
plt.figure(figsize=(10 ,4))
sns.histplot(New_df[New_df['Target'] == 0]['Total letters'], color='green', alpha=0.7, label='Target 0')
sns.histplot(New_df[New_df['Target'] == 1]['Total letters'], color='red', alpha=0.5, label='Target 1')
plt.show


# In[33]:


plt.figure(figsize=(10 ,4))
sns.histplot(New_df[New_df['Target'] == 0]['Total word'], color='green', alpha=0.7, label='Target 0')
sns.histplot(New_df[New_df['Target'] == 1]['Total word'], color='red', alpha=0.5, label='Target 1')
plt.show


# In[34]:


plt.figure(figsize=(10 ,4))
sns.histplot(New_df[New_df['Target'] == 0]['Total Sentence'], color='green', alpha=0.7, label='Target 0')
sns.histplot(New_df[New_df['Target'] == 1]['Total Sentence'], color='red', alpha=0.5, label='Target 1')
plt.show


# In[35]:


sns.pairplot(New_df, hue='Target')


# In[36]:


sns.heatmap(New_df.corr(),annot=True)


# # Data Preprocessing

# In[37]:


# first we convert senetences in lower case
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
nltk.download('stopwords')
import string
def Transform_text(Text):
    Text = Text.lower()
    Text = nltk.word_tokenize(Text)
    
    y = []
    for i in Text:
       if i.isalnum():
            y.append(i)
            
    Text = y[:]
    y.clear()
            
    for i in Text:
      if i not in stopwords.words('english') and i not in string.punctuation:
                    y.append(i)
    Text = y[:]
    y.clear()
    
           
    for i in Text:
        y.append(ps.stem(i))
    return "  ".join(y)


# In[38]:


New_df['Transform_text'] = New_df["Text"].apply(Transform_text)


# In[39]:


New_df.head()


# In[40]:


# Print the list of column names in your DataFrame
print(New_df.columns)


# In[43]:


#for hamm msgs word cloud
plt.figure(figsize=(15,6))
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
ham_wc = wc.generate(New_df[New_df["Target"] == 0] ['Transform_text'].str.cat(sep=" "))
plt.imshow(ham_wc)


# In[44]:


#for spam msgs word cloud
from wordcloud import WordCloud 
plt.figure(figsize=(15,6))
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
spam_wc = wc.generate(New_df[New_df["Target"] == 1] ['Transform_text'].str.cat(sep=" "))
plt.imshow(spam_wc)


# In[45]:


spam_corpus = []
for msg in New_df[New_df['Target'] == 1]['Transform_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[46]:


len(spam_corpus)


# In[47]:


from collections import Counter
plt.figure(figsize=(15,6))
sns.barplot(x = pd.DataFrame(Counter(spam_corpus).most_common(30))[0],y = pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation ='vertical')
plt.show


# In[48]:


ham_corpus = []
for msg in New_df[New_df['Target'] == 0]['Transform_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[49]:


len(ham_corpus)


# In[50]:


plt.figure(figsize=(15,6))
sns.barplot(x = pd.DataFrame(Counter(ham_corpus).most_common(30))[0],y = pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation ='vertical')
plt.show


# # Model Building

# In[51]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report

New_df['Transform_text'] = New_df["Text"].apply(Transform_text)
X = New_df['Transform_text'].values
y = New_df['Target'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Define a list of models with their hyperparameter grids for GridSearchCV
models = [
    {
        'name': 'Multinomial Naive Bayes',
        'model': MultinomialNB(),
        'params': {}
    },
    {
        'name': 'Support Vector Machine',
        'model': SVC(),
        'params': {'C': [1, 10, 100], 'kernel': ['linear', 'rbf']}
    },
    {
        'name': 'Logistic Regression',
        'model': LogisticRegression(),
        'params': {'C': [0.1, 1, 10]}
    },
    {
        'name': 'Random Forest',
        'model': RandomForestClassifier(),
        'params': {'n_estimators': [50, 100, 200]}
    },
    {
        'name': 'XGBoost',
        'model': xgb.XGBClassifier(),
        'params': {'learning_rate': [0.1, 0.2], 'max_depth': [3, 4]}
    }
]

best_model = None
best_score = 0

# Iterate through models, perform GridSearchCV, and find the best model
for model_info in models:
    model_name = model_info['name']
    model = model_info['model']
    params = model_info['params']

    gs = GridSearchCV(model, params, cv=5, verbose=1)
    gs.fit(X_train_tfidf, y_train)

    if gs.best_score_ > best_score:
        best_score = gs.best_score_
        best_model = gs.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test_tfidf)
report = classification_report(y_test, y_pred)
print("Best Model:", best_model)
print("Best Model's Classification Report:\n", report)

