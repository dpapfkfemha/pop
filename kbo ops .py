#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
plt.style.use('fivethirtyeight') #파이브써티에잇 www.fivethirtyeight.com
import warnings
warnings.filterwarnings('ignore') #워닝 무시


# In[3]:


os.chdir('C:/Users/user/Desktop/2020취업준비/KBO OPS/6th_data')


# In[4]:


regular=pd.read_csv("Regular_Season_Batter.csv")


# In[5]:


regular.head()


# In[6]:


regular=regular.loc[~regular['OPS'].isnull(),]
regular.info()


# In[7]:


agg={}
for i in regular.columns:
    agg[i]=[]


# In[8]:


submission=pd.read_csv("submission.csv") 


# In[9]:


submission.head()


# In[10]:


for i in submission['batter_name'].unique():
    for j in regular.columns:
        if j in ['batter_id', 'batter_name','height/weight','year_born','position','starting_salary']:
            agg[j].append(regular.loc[regular['batter_name']==i,j].iloc[0])
        elif j=='year':
            agg[j].append(2019)
        else:
            agg[j].append(0)


# In[67]:


pd.DataFrame(agg)


# In[12]:


regular=pd.concat([regular, pd.DataFrame(agg)])


# In[13]:


pd.set_option('display.max_columns',500)
regular.head(-15)


# In[15]:


regular.columns


# In[16]:


corr = regular.loc[:,regular.dtypes == 'float64'].corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))


# In[17]:


corr2 = regular.loc[:,regular.dtypes == 'int64'].corr()
sns.heatmap(corr2, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))


# In[18]:


sns.pairplot(regular.loc[:,regular.dtypes == 'float64'])


# In[19]:


sns.distplot(regular['year'])


# 년도가 줄어들수록 그 수가 작다

# In[20]:


regular['year'].describe()


# In[21]:


sns.distplot(regular['AB'])


# 타석에 100타석 이하로 들어온 타자들이 꽤 많다

# In[22]:


sns.distplot(regular['OPS'].dropna())


# ops 는 대략적으로 정규분포를 따른다

# In[23]:


plt.scatter(regular['AB'],regular['OPS'])
plt.xlabel('AB')
plt.ylabel('OPS')


# 타석수와 OPS는 양의 상관관계를 가진다. 시즌이 진행됨에 따라 잘하는 선수는 많이 기용되고, 못하는 선수는 적게 기용되기 때문이라고 보여짐

# In[24]:


regular.groupby('position')['OPS'].mean()


# # 선수의 운과 진짜 실력 구분하기

# In[25]:


def get_self_corr(var, regular=regular):
    x=[]
    y=[]
    regular1=regular.loc[regular['AB']>=50,]
    for name in regular1['batter_name'].unique():
        a=regular1.loc[regular1['batter_name']==name,].sort_values('year')
        k=[]
        for i in a['year'].unique():
            if (a['year']==i+1).sum()==1:
                k.append(i)
        for i  in k:
            x.append(a.loc[a['year']==i, var].iloc[0])
            y.append(a.loc[a['year']==i+1, var].iloc[0])
        plt.scatter(x,y)
        plt.title(var)
        plt.show()
        print(pd.Series(x).corr(pd.Series(y))**2)
        
    regular['1B']=regular['H']-regular['2B']-regular['3B']-regular['HR']


# In[27]:


b=regular1.loc[regular1['batter_name']=='고동진', ].sort_values('year')
b


# In[ ]:





# In[ ]:





# In[31]:


for i in ['HR','BB']:
    get_self_corr(i)


# In[32]:


regular['1b_luck']=regular['1B']/(regular['AB']-regular['HR']-regular['SO'])
regular['2b_luck']=regular['2B']/(regular['AB']-regular['HR']-regular['SO'])
regular['3b_luck']=regular['3B']/(regular['AB']-regular['HR']-regular['SO'])


# In[33]:


for j in ['avg', 'G', 'AB', 'R', 'H','2B', '3B', 'HR', 'TB', 'RBI', 
          'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP','SLG', 'OBP', 'E','1b_luck','2b_luck','3b_luck']:
    
    lag_1_avg=[]
    for i in range(len(regular)): 
        if len(regular.loc[(regular['batter_name']==regular['batter_name'].iloc[i])
                           &(regular['year']==regular['year'].iloc[i]-1)][j])==0:
            lag_1_avg.append(np.nan)
        else:
            lag_1_avg.append(regular.loc[(regular['batter_name']==regular['batter_name'].iloc[i])
                                         &(regular['year']==regular['year'].iloc[i]-1)][j].iloc[0])
    
    regular['lag_1_'+j]=lag_1_avg
    print(j)


# In[34]:


regular.head()


# In[36]:


def get_nujuk(name,year,var):
    if (len(regular.loc[
                        (regular['batter_name']==name)
                        &(regular['year']<year-1)
                        ,'H']
                        )!=0):
        return regular.loc[(regular['batter_name']==name)&(regular['year']<year-1),var].sum()
    else:
        return np.nan

for i in ['G', 'AB', 'R', 'H','2B', '3B', 'HR', 'TB', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO']:
    regular['total_'+i]=regular.apply(lambda x: get_nujuk(x['batter_name'],x['year'],i),axis=1)


# In[45]:


regular.head()


# In[40]:


regular.loc[regular['batter_name']=='최정']


# In[41]:


from sklearn.ensemble import RandomForestRegressor


# In[42]:


train=regular.loc[regular['year']<=2017,]
test=regular.loc[regular['year']==2018,]
y_train=train['OPS']
X_train=train[[x for x in regular.columns if ('lag' in x)|('total' in x)]]  

y_test=test['OPS']
X_test=test[[x for x in regular.columns if ('lag' in x)|('total' in x)]]


# In[44]:


X_train.head()


# In[47]:


X_train.fillna(-1).head()


# In[48]:


rf=RandomForestRegressor(n_estimators=500)
rf.fit(X_train.fillna(-1),y_train,sample_weight=train['AB'])


# In[50]:


pred=rf.predict(X_test.fillna(-1))


# In[51]:


real=test['OPS']
ab=test['AB']

from sklearn.metrics import mean_squared_error
mean_squared_error(real,pred,sample_weight=ab)**0.5


# In[53]:


train=regular.loc[regular['year']<=2018,]
test=regular.loc[regular['year']==2019,]
y_train=train['OPS']
X_train=train[[x for x in regular.columns if ('lag' in x)|('total' in x)]]


rf=RandomForestRegressor(n_estimators=500)
rf.fit(X_train.fillna(-1),y_train,sample_weight=train['AB'])


# In[54]:


test=regular.loc[regular['year']==2019,]


# In[55]:


pred=rf.predict(test[[x for x in regular.columns if ('lag' in x)|('total' in x)]].fillna(-1))


# In[57]:


result=pd.DataFrame({'batter_id':test['batter_id'],'OPS':pred})


# In[64]:


pd.DataFrame({'batter_name':test['batter_name'],'OPS':pred}).to_csv("baseline_submission.csv",index=False)


# In[66]:


result=pd.read_csv('baseline_submission.csv')
result.head(30)


# In[ ]:




