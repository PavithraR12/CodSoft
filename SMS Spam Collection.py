#!/usr/bin/env python
# coding: utf-8

# # Import necessary libraries

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# # Load the dataset

# In[13]:


url = 'C:/Users/Asus/Downloads/archive (2).zip'
df = pd.read_csv(url, encoding='latin-1')
df = df[['v1', 'v2']] 
df


# # Preprocess the data

# In[16]:


df['v1'] = df['v1'].map({'ham': 0, 'spam': 1})
df


# # Split the data into training and testing sets

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(df['v2'], df['v1'], test_size=0.2, random_state=42)


# # Create a Bag-of-Words model using CountVectorizer

# In[15]:


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
vectorizer


# # Train a Naive Bayes classifier

# In[9]:


classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)


# # Make predictions on the test set

# In[17]:


predictions = classifier.predict(X_test_vec)
predictions


# # Evaluate the model

# In[11]:


accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)


# # Results

# In[12]:


print(f'Accuracy: {accuracy}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)


# In[ ]:




