#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import array
from sklearn.neighbors import KNeighborsClassifier


# In[98]:


data = pd.read_csv('../train.csv')
test = pd.read_csv('../test.csv')


# In[100]:


Descr = [0] * len(data.index)

for n in range(len(data.index)):
    
 text = str(data.loc[n,['Description']])
 text = text.lower()
 if(text.find('good')!=-1 or text.find('cute')!=-1 or text.find('healthy')!=-1 or text.find('playful')!=-1 or 
    text.find('spayed')!=-1 or text.find('neutered')!=-1 or text.find('active')!=-1 or text.find('Good')!=-1 
    or text.find('cutie')!=-1 or text.find('calm')!=-1 or text.find('quiet')!=-1 or text.find('trained')!=-1
    or text.find('good temperement')!=-1 or text.find('Love human')!=-1 or text.find('well behaved')!=-1 
    or text.find('feisty')!=-1 or text.find('obey')!=-1 or text.find('domesticated')!=-1 or text.find('joyful')!=-1
    or text.find('cuddly')!=-1 or text.find('loyal')!=-1 or text.find('adorable')!=-1 or text.find('independent')!=-1
    or text.find('toilet trained')!=-1 or text.find('not lazy')!=-1 or text.find('Toilet Trained')!=-1):
        Descr[n] = 1
 if(text.find('not good')!=-1 or text.find('not cute')!=-1 or text.find('not healthy')!=-1 or text.find('playful')!=-1 or 
    text.find('not spayed')!=-1 or text.find('not neutered')!=-1 or text.find('not active')!=-1 or text.find('Good')!=-1 
    or text.find('not calm')!=-1 or text.find('not quiet')!=-1 or text.find('not trained')!=-1
    or text.find('not good temperement')!=-1 or text.find('scared of human')!=-1 or text.find('badly behaved')!=-1 
    or text.find('not vaccinated')!=-1 or text.find('not obey')!=-1 or text.find('not domesticated')!=-1 or text.find('sad')!=-1
    or text.find('steals')!=-1 or text.find('not loyal')!=-1 or text.find('dependent')!=-1 or text.find('not independent')!=-1
    or text.find('not toilet trained')!=-1 or text.find('Not Toilet Trained')!=-1 or text.find('Toilet Trained')!=-1 
    or text.find('Not Toilet trained')!=-1 or text.find('unhealthy')!=-1 or text.find('lazy')!=-1 or text.find('attackes')!=-1
   or text.find('assault')!=-1):
        Descr[n] = 0  
        
data['Describe'] = Descr


# In[101]:


data['Describe'] = Descr


# In[102]:


x = data.loc[:,['Type','Age','Breed1','Breed2','Gender','Color1','Color2','Color3','MaturitySize','FurLength','Vaccinated'
                ,'Dewormed','Sterilized','Health','Quantity','Fee','State','VideoAmt','PhotoAmt','Describe']]
y = data['AdoptionSpeed']


# In[103]:


lmod = linear_model.LogisticRegression()

lmod.fit(x,y)
pred = lmod.predict(x)
rsquare = r2_score(y,pred)
prob1 = lmod.predict_proba(x)


# In[104]:


lmod1df = pd.DataFrame(prob1)
lmod1df.columns = ['L_0','L_1','L_2','L_3','L_4']
data = pd.concat([data,lmod1df],axis=1)


# In[105]:


rfmod = RandomForestClassifier(n_estimators = 50 , n_jobs = 500)
rfmod.fit(x,y)
pred = rfmod.predict(x)
rquare = r2_score(y,pred)
prob2 = rfmod.predict_proba(x)


# In[106]:


rfmoddf = pd.DataFrame(prob2)
rfmoddf.columns = ['R_0','R_1','R_2','R_3','R_4']
data = pd.concat([data,rfmoddf],axis=1)


# In[107]:


treemod = DecisionTreeClassifier()
treemod.fit(x,y)
pred = treemod.predict(x)
rquare = r2_score(y,pred)
prob3 = treemod.predict_proba(x)


# In[108]:


treemoddf = pd.DataFrame(prob3)
treemoddf.columns = ['T_0','T_1','T_2','T_3','T_4']

data = pd.concat([data,treemoddf],axis=1)


# In[109]:


knnmod = KNeighborsClassifier(n_neighbors=3)
knnmod.fit(x,y)
prob4 = knnmod.predict_proba(x)


# In[110]:


knnmoddf = pd.DataFrame(prob4)
knnmoddf.columns = ['K_0','K_1','K_2','K_3','K_4']
data = pd.concat([data,knnmoddf],axis=1)


# In[111]:


x = data.loc[:,['Type','Age','Breed1','Breed2','Gender','Color1','Color2','Color3','MaturitySize','FurLength','Vaccinated'
                ,'Dewormed','Sterilized','Health',
                'Quantity','Fee','State','VideoAmt','PhotoAmt','Describe','L_1','L_2','L_3','L_0','L_4','R_0','R_1','R_2'
               ,'R_3','R_4','T_0','T_1','T_2','T_3','T_4','K_0','K_1','K_2','K_3','K_4']]
y = data['AdoptionSpeed']


# In[112]:


xgbmod = XGBClassifier(learning_rate =0.3,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
xgbmod.fit(x,y)
pred = xgbmod.predict(x)
pred.round()
rquare = r2_score(y,pred)
rsquare


# In[113]:


Descr = [0] * len(test.index)

for n in range(len(test.index)):
    
 text = str(test.loc[n,['Description']])
 text = text.lower()
 if(text.find('good')!=-1 or text.find('cute')!=-1 or text.find('healthy')!=-1 or text.find('playful')!=-1 or 
    text.find('spayed')!=-1 or text.find('neutered')!=-1 or text.find('active')!=-1 or text.find('Good')!=-1 
    or text.find('cutie')!=-1 or text.find('calm')!=-1 or text.find('quiet')!=-1 or text.find('trained')!=-1
    or text.find('good temperement')!=-1 or text.find('Love human')!=-1 or text.find('well behaved')!=-1 
    or text.find('feisty')!=-1 or text.find('obey')!=-1 or text.find('domesticated')!=-1 or text.find('joyful')!=-1
    or text.find('cuddly')!=-1 or text.find('loyal')!=-1 or text.find('adorable')!=-1 or text.find('independent')!=-1
    or text.find('toilet trained')!=-1 or text.find('Toilet trained')!=-1 or text.find('Toilet Trained')!=-1):
        Descr[n] = 1
 if(text.find('not good')!=-1 or text.find('not cute')!=-1 or text.find('not healthy')!=-1 or text.find('playful')!=-1 or 
    text.find('not spayed')!=-1 or text.find('not neutered')!=-1 or text.find('not active')!=-1 or text.find('Good')!=-1 
    or text.find('not calm')!=-1 or text.find('not quiet')!=-1 or text.find('not trained')!=-1
    or text.find('not good temperement')!=-1 or text.find('scared of human')!=-1 or text.find('badly behaved')!=-1 
    or text.find('not vaccinated')!=-1 or text.find('not obey')!=-1 or text.find('not domesticated')!=-1 or text.find('sad')!=-1
    or text.find('steals')!=-1 or text.find('not loyal')!=-1 or text.find('dependent')!=-1 or text.find('not independent')!=-1
    or text.find('not toilet trained')!=-1 or text.find('Not Toilet Trained')!=-1 or text.find('Toilet Trained')!=-1 
    or text.find('Not Toilet trained')!=-1 or text.find('unhealthy')!=-1 or text.find('lazy')!=-1 or text.find('attackes')!=-1
   or text.find('assault')!=-1):
        Descr[n] = 0  
        
test['Describe'] = Descr


# In[114]:


xtest1 = test.loc[:,['Type','Age','Breed1','Breed2','Gender','Color1','Color2','Color3','MaturitySize','FurLength','Vaccinated'
                ,'Dewormed','Sterilized','Health',
                'Quantity','Fee','State','VideoAmt','PhotoAmt','Describe','L_1','L_2','L_3','L_0','L_4','R_0','R_1','R_2'
               ,'R_3','R_4','T_0','T_1','T_2','T_3','T_4','K_0','K_1','K_2','K_3','K_4']]
ytest = xgbmod.predict(xtest1)
Submission = pd.DataFrame()
Submission['PetID'] = test['PetID']
Submission['AdoptionSpeed'] = ytest
Submission
#Submission.to_csv('submission.csv',index = None)
