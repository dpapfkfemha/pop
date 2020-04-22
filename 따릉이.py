#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
os.chdir("C:/Users/user/Desktop/2020취업준비/ML스터디/hw 4주")


# In[4]:


train=pd.read_csv('train (1).csv')
test=pd.read_csv('test (1).csv')


# In[5]:


train.head(10)


# In[6]:


train.rename(columns={'hour_bef_temperature':'temperature','hour_bef_precipitation':'precipitation',
                      'hour_bef_windspeed':'windspeed', 'hour_bef_humidity':'humidity',
                      'hour_bef_visibility': 'visibility', 'hour_bef_ozone': 'ozone',
                      'hour_bef_pm10':'pm10','hour_bef_pm2.5':'pm25'}, inplace=True)

train=train.iloc[:,0:11]
train.head()


# In[7]:


test.rename(columns={'hour_bef_temperature':'temperature','hour_bef_precipitation':'precipitation',
                      'hour_bef_windspeed':'windspeed', 'hour_bef_humidity':'humidity',
                      'hour_bef_visibility': 'visibility', 'hour_bef_ozone': 'ozone',
                      'hour_bef_pm10':'pm10','hour_bef_pm2.5':'pm25'}, inplace=True)
test.head()


# In[8]:


train=train.dropna()


# In[9]:


sns.distplot(train['temperature'])


# In[10]:


plt.scatter(train['temperature'], train['count'])


# In[11]:


sns.distplot(train['windspeed'])


# In[13]:


print(train.precipitation.value_counts())


# In[12]:


sns.pairplot(train[['count','temperature', 'windspeed', 'humidity', 'visibility', 'ozone', 'pm10', 'pm25']])


# In[15]:


from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-white')


# In[18]:


model=smf.ols(formula = 'count ~ temperature+precipitation+windspeed+humidity+ozone+pm10', data=train)  ##pm25는 pm10과 유사한 변수 포함되도 괜찮
result=model.fit()
result.summary() 


# In[19]:


model=smf.ols(formula = 'count ~ temperature+precipitation+windspeed+humidity+visibility+ozone+pm10', data=train)  
result=model.fit()
result.summary() 


# In[29]:


train_x =train.drop('count', axis=1)
train_y =train['count']
train_y.head()


# In[31]:


test.head()


# In[32]:


# modules
import warnings
warnings.filterwarnings("ignore") #경고메시지 방지를 위한 명령어

import pandas as pd  ##데이터 처리
import numpy as np  ##계산을 주로 다루는 라이브러리
import seaborn as sns ##통계적인 시각화
import matplotlib.pyplot as plt #시각화

from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import make_scorer, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# In[33]:


xgb_model = XGBRegressor()     ##3개 입력하면 각 경우의 수를 보여줌
xgb_params = {
    'n_estimators' : [500, 1000, 3000], 
    'eta' : [0.01, 0.1],   ##learing rate 트리들이 만들어낸 예측값이 얼마나 적용되는 지와 관련, overfitting 방지
    'max_depth' : [5, 10], ##tree 가 가질 수 있는 깊이, 과적합 방지
    'lambda' : [0, 1, 3],  
    'min_child_weight' : [0, 1, 3] 
    
    ##subsample: training set에서 얼마나 쓸거냐
    ##gamma: ##tree에 leaf node에 수에 곱해주는 값, 감마가 커지면 tree leaf가 많아지는 거 방지, tree 구조가 너무 복잡해지지 않도록 함
}


# In[34]:


def rmse(real, pred):
    return np.sqrt(np.mean(np.square(real - pred)))


# In[35]:


reg = GridSearchCV(xgb_model, xgb_params, scoring=make_scorer(rmse, greater_is_better=False), cv=5)
##1. 최적의 파라미터를 찾아주고, 2. 교차검증도 해준다!
reg.fit(train_x, train_y) #xgboost 실행


# In[36]:


reg.best_score_


# In[37]:


reg.best_params_


# In[38]:


reg.best_estimator_


# 3. submission

# In[53]:


pred=pd.DataFrame({'id': test['id'], 'count': reg.predict(test)})
pred.to_csv('submission.csv', index=False) ###디렉토리를 설정해줘야 저장할 때도 편하네


# In[52]:


pred.head(15)


# In[45]:


test.head(15)


# In[46]:


test['id']


# In[ ]:




