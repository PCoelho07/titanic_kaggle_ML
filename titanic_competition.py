
# coding: utf-8

# In[81]:


import pandas as pd
import numpy as np

test_df = pd.read_csv('test.csv')
train_df = pd.read_csv('train.csv')

train_df.head()


# In[82]:


print train_df.columns.values


# In[83]:


train_df.head()


# In[84]:


train_df.tail()


# In[85]:


train_df.info() 
print ("_"*40)
test_df.info()


# In[86]:


train_df.describe()


# In[87]:


train_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
train_df.head()


# In[88]:


new_train_df = pd.get_dummies(train_df)
new_test_df = pd.get_dummies(test_df)
new_train_df.head()


# In[89]:


new_train_df.describe()


# In[90]:


new_train_df.columns.values


# In[91]:


for item in new_train_df:
    print item


# In[92]:


new_train_df.isnull().sum().sort_values(ascending=False).head(10)


# In[93]:


new_train_df['Age'].fillna(new_train_df['Age'].mean(), inplace=True)
new_test_df['Age'].fillna(new_test_df['Age'].mean(), inplace=True)


# In[94]:


new_test_df.isnull().sum().head(10)


# In[95]:


new_test_df['Fare'].fillna(new_test_df['Fare'].mean(), inplace=True)


# In[96]:


X = new_train_df.drop(['Survived'], axis=1)
y = new_train_df['Survived']


# In[97]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=3, random_state=0)
clf.fit(X, y)


# In[98]:


clf.score(X, y)


# In[103]:


submission = pd.DataFrame()
submission['PassengerId'] = new_test_df['PassengerId']
submission['Survived'] = clf.predict(new_test_df)


# In[104]:


submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




