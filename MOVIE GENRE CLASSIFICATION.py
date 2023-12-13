#!/usr/bin/env python
# coding: utf-8

# In[22]:


get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install pandas')
get_ipython().system('pip install nltk')


# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import re
import nltk
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# In[4]:


dataset_path = "C:/Users/Asus/Downloads/train_data.txt"
df = pd.read_csv(dataset_path,sep = ':::', names =['Title','Genre','Description'], engine ='python')
df


# In[5]:


data1 = "C:/Users/Asus/Downloads/test_data.txt"
data_set1 = pd.read_csv(data1, sep=':::', names=['Id', 'Title', 'Description'], engine='python')
data_set1.head()


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(14,7))
sns.countplot(data=df,y='Genre',order = df['Genre'].value_counts().index,palette='magma')
plt.xlabel('Count',fontsize=12,fontweight='bold')
plt.ylabel('Count',fontsize=12,fontweight='bold')


# In[30]:


plt.figure(figsize=(14,7))
counts= df['Genre'].value_counts()
sns.barplot(x=counts.index, y=counts, palette='viridis')
plt.xlabel('Genre',fontsize=12,fontweight = 'bold')
plt.ylabel('count',fontsize=12,fontweight = 'bold')
plt.title('Distribution of Genres',fontsize=14,fontweight = 'bold')
plt.xticks(rotation=90,fontsize=13,fontweight = 'bold')
plt.show()


# In[1]:


import nltk
nltk.download('stopwords')


# In[2]:


nltk.download('punkt')


# In[21]:


stemmer = LancasterStemmer()
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\S+','',text)
    text = re.sub(r'http\S+','',text)
    text = re.sub(r'pic.\S+','',text)
    text = re.sub(r"[^a-zA-Z+']",' ',text)
    text = re.sub(r'\s+[a-zA-Z]\s+',' ',text+' ')
    text="".join([i for i in text if i not in string.punctuation])
    words=nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    text=" ".join([i for i in words if i not in stopwords and len(i)>2])
    text=re.sub("\s[\s]+"," ",text).strip()
    return text
df['Text_cleaning']=df['Description'].apply(clean_text)
data_set1['Text_cleaning']=data_set1['Description'].apply(clean_text)


# In[25]:


tfidf_vectorizer = TfidfVectorizer()
X_train =  tfidf_vectorizer.fit_transform(df['Text_cleaning'])
X_test =  tfidf_vectorizer.transform(data_set1['Text_cleaning'])
X = X_train
y = df['Genre']
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.25,random_state = 42)

classifier = MultinomialNB()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_val)

accuracy = accuracy_score(y_val,y_pred)
print("Validation_accuracy:",accuracy)
print(classification_report(y_val,y_pred))


# In[26]:


X_test_prediction = classifier.predict(X_test)
data_set1['Predicted Genre'] = X_test_prediction


# In[16]:


data_set1.to_csv('predicted_genres.csv',index = False)
print(data_set1)


# In[ ]:




