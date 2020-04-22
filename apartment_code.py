#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 


# In[2]:


train = pd.read_csv('C:/Users/choongwon/Desktop/apartment/train.csv', encoding='utf-8')
park = pd.read_csv('C:/Users/choongwon/Desktop/apartment/park.csv', encoding='utf-8')
day_care_center = pd.read_csv('C:/Users/choongwon/Desktop/apartment/day_care_center.csv', encoding='utf-8')
test = pd.read_csv('C:/Users/choongwon/Desktop/apartment/test.csv', encoding='utf-8')


# In[3]:


train.info() # train 데이터 information
train.isnull().sum() # train 데이터 결측치의 개수 > 결측치가 없다!


# In[5]:


park.info() # park 데이터 information
park.isnull().sum() # park 데이터 결측치의 개수


# In[6]:


day_care_center.info() # day_care_center 데이터 information
day_care_center.isnull().sum() # day_care_center 데이터 결측치의 개수


# In[7]:


# 같은 동에 있는 park 갯수, 그리고 그 park의 수준에 따라 또 나뉠듯
# 같은 구에 있는 park 갯수
# 같은 구에 있는 day_care_center 갯수 를 새로운 변수로 둘 수도 있을듯
# 물어볼것 : day_care_type과 같이 categorical data를 regression에 사용할 때 dummy variable을 사용해야 하는가? 7갠데?!
# 중점적으로 해아할것 : 데이터를 합쳐야 하는데 어떤 방식으로 합쳐야할지?


# In[8]:


# 생각해볼것
# 1. train 데이터에 park와 day_care_center를 합쳐야 하는데 어떤 방식으로 합칠지? 동기준? 구 기준? 같은 동에 여러개의 park가 있으면 어떻게 처리를 할지? 근처에 공원이 없으면 어떤 식으로 처리를 해야할지?
# 2. domain 지식으로 날려야 할 변수와 살려야 할 변수를 생각해봐야함 결측치가 많은 데이터를 죽일지 살릴지, 살리면 어떤 방식으로 살려볼지?
# 3. catagorical data, 시간데이터(ex: 년월일)는 어떻게 처리해야하는지???
# 4. 데이터 스케일링?


# In[9]:


train1 = train.iloc[:, [0, 1, 2, 3, 6, 7, 8, 9, 11, 12]]
train1['model_year'] = train1['transaction_year_month'] // 100 - train1['year_of_completion']
train1.head()


# In[10]:


train1 = train1.iloc[:,[0, 1, 2, 3, 4, 5, 10, 7, 8, 9]]
train1


# In[11]:


park1 = park.iloc[:,[0, 1, 2, 3, 4, 5, 11]]
park1.head()
# 가장 큰 면적 column 한개, 공원의 총 갯수 한개


# In[12]:


train1_seoul = train1[train1.city == '서울특별시']
train1_busan = train1[train1.city == '부산광역시']


# In[13]:


park1_seoul = park1[park1.city == '서울특별시']
park1_busan = park1[park1.city == '부산광역시']


# In[14]:


pd.value_counts(train1['addr_kr'], sort = True)


# In[15]:


pd.value_counts(day_care_center['day_care_type'])


# In[16]:


day_care_center['day_care_type']


# In[17]:


tp = {'가정' : 0, '민간' : 1, '국공립' : 2, '직장' : 3, '법인·단체' : 4, '사회복지법인' : 5, '협동' : 6}
day_care_center['day_care_type'] = day_care_center['day_care_type'].map(tp)


# In[18]:


day_care_center.head()


# In[19]:


plt.bar(day_care_center['day_care_type'], day_care_center['day_care_baby_num'])
plt.show()


# In[20]:


plt.bar(day_care_center['day_care_type'], day_care_center['teacher_num'])
plt.show()


# In[21]:


day_care_center['baby_teacher_ratio'] = day_care_center['day_care_baby_num'] / day_care_center['teacher_num']
day_care_center.head()


# In[22]:


pd.pivot_table(day_care_center, index = ['day_care_type'])


# In[23]:


day_care_center.loc[(day_care_center['day_care_type'] == 0) & (day_care_center['teacher_num'].isnull()), ['teacher_num']] = 5
day_care_center.loc[(day_care_center['day_care_type'] == 1) & (day_care_center['teacher_num'].isnull()), ['teacher_num']] = 11
day_care_center.loc[(day_care_center['day_care_type'] == 2) & (day_care_center['teacher_num'].isnull()), ['teacher_num']] = 12
day_care_center.loc[(day_care_center['day_care_type'] == 3) & (day_care_center['teacher_num'].isnull()), ['teacher_num']] = 15
day_care_center.loc[(day_care_center['day_care_type'] == 4) & (day_care_center['teacher_num'].isnull()), ['teacher_num']] = 10
day_care_center.loc[(day_care_center['day_care_type'] == 5) & (day_care_center['teacher_num'].isnull()), ['teacher_num']] = 14
day_care_center.loc[(day_care_center['day_care_type'] == 6) & (day_care_center['teacher_num'].isnull()), ['teacher_num']] = 7


# In[24]:


day_care_center.loc[(day_care_center['day_care_type'] == 0) & (day_care_center['teacher_num'] == 0), ['teacher_num']] = 5
day_care_center.loc[(day_care_center['day_care_type'] == 1) & (day_care_center['teacher_num'] == 0), ['teacher_num']] = 11
day_care_center.loc[(day_care_center['day_care_type'] == 2) & (day_care_center['teacher_num'] == 0), ['teacher_num']] = 12
day_care_center.loc[(day_care_center['day_care_type'] == 3) & (day_care_center['teacher_num'] == 0), ['teacher_num']] = 15
day_care_center['baby_teacher_ratio'] = day_care_center['day_care_baby_num'] / day_care_center['teacher_num']
pd.pivot_table(day_care_center, index = ['day_care_type'])


# In[25]:


plt.scatter(day_care_center['day_care_baby_num'], day_care_center['CCTV_num'], s = 10)


# In[26]:


plt.scatter(day_care_center['day_care_baby_num'], day_care_center['playground_num'], s = 10)


# In[30]:


pd.value_counts(test['addr_kr'], sort = True)


# In[31]:


park.head()


# In[32]:


day_care_center.head()


# In[67]:


day_care_center.loc[(day_care_center['day_care_baby_num'] < 30)]


# In[68]:


pd.value_counts(day_care_center.loc[(day_care_center['day_care_baby_num'] < 30)], sort = True)


# In[69]:


pd.value_counts(day_care_center.loc[(day_care_center['day_care_baby_num'] < 30)]['gu'], sort = True)


# In[70]:


pd.value_counts(day_care_center.loc[(day_care_center['day_care_baby_num'] < 60) & (day_care_center['day_care_baby_num'] >= 30)]['gu'], sort = True)


# In[71]:


pd.value_counts(day_care_center.loc[(day_care_center['day_care_baby_num'] >= 60)]['gu'], sort = True)


# In[80]:


day_care_center.loc[(day_care_center['day_care_baby_num'] < 30), ['reference_date']] = 'C'
day_care_center.loc[(day_care_center['day_care_baby_num'] >= 30) & (day_care_center['day_care_baby_num'] < 60), ['reference_date']] = 'B'
day_care_center.loc[(day_care_center['day_care_baby_num'] >= 60), ['reference_date']] = 'A'


# In[81]:


day_care_center.head()


# In[82]:


pd.value_counts(day_care_center['gu'], sort = True)

day_care_center['gu', 'reference_date']
# In[88]:


day_care_center_grade = day_care_center.loc[:, ['gu', 'reference_date']]


# In[90]:


day_care_center_grade.head()


# In[99]:


day_care_center_grade['gu'].unique()


# In[104]:


day_care_center_new = {'구' : day_care_center_grade['gu'].unique()}


# In[106]:


day_care_center_new = pd.DataFrame(day_care_center_new)


# In[108]:


day_care_center_new.head


# In[122]:


num_A = pd.value_counts(day_care_center_grade.loc[(day_care_center_grade['reference_date'] == 'A'), ['gu']]['gu'])


# In[123]:


num_B = pd.value_counts(day_care_center_grade.loc[(day_care_center_grade['reference_date'] == 'B'), ['gu']]['gu'])


# In[124]:


num_C = pd.value_counts(day_care_center_grade.loc[(day_care_center_grade['reference_date'] == 'C'), ['gu']]['gu'])


# In[129]:


num_A = pd.DataFrame(num_A)


# In[130]:


num_B = pd.DataFrame(num_B)
num_C = pd.DataFrame(num_C)


# In[145]:


num_A.reset_index(inplace = True)
num_B.reset_index(inplace = True)
num_C.reset_index(inplace = True)


# In[146]:


num_A.sort_values(by='index', ascending = True, inplace = True)
num_B.sort_values(by='index', ascending = True, inplace = True)
num_C.sort_values(by='index', ascending = True, inplace = True)


# In[147]:


num_A.head()


# In[148]:


num_B.head()


# In[149]:


num_C.head()


# In[152]:


k = pd.merge(num_A, num_B, on = 'index')


# In[153]:


num = pd.merge(k, num_C, on = 'index')


# In[154]:


num.head()


# In[155]:


num['per_A'] = num['gu_x'] / (num['gu_x'] + num['gu_y'] + num['gu'])
num['per_B'] = num['gu_y'] / (num['gu_x'] + num['gu_y'] + num['gu'])
num['per_C'] = num['gu'] / (num['gu_x'] + num['gu_y'] + num['gu'])


# In[156]:


num.head()


# In[157]:


num.drop(['gu_x', 'gu_y', 'gu'], axis='columns', inplace=True)
num.head()


# In[158]:


train2 = pd.read_csv('C:/Users/choongwon/Desktop/apartment/train2.csv', encoding='utf-8')
train2.head()


# In[160]:


gu_dong = park.loc[:, ['gu', 'dong']]
gu_dong.head()


# In[166]:


gu_dong.dropna(inplace = True) 


# In[167]:


gu_dong.isnull().sum()


# In[186]:


train3 = pd.merge(train2, gu_dong, on = 'dong', how = 'inner')


# In[187]:


train3.isnull().sum()


# In[189]:


gu_dong.info()


# In[190]:


gu_dong.drop_duplicates(inplace = True)


# In[191]:


gu_dong.info()

train3 = pd.merge(train2, gu_dong, on = 'dong', how = 'left')
# In[192]:


train3 = pd.merge(train2, gu_dong, on = 'dong', how = 'left')
train3.info()


# In[193]:


train3.isnull().sum()


# In[184]:


train3.count()


# In[185]:


train2.count()


# In[194]:


num.rename(columns={num.columns[0] : 'gu'}, inplace = True)
num.head()


# In[195]:


train4 = pd.merge(train3, num, how = 'left', on = 'gu')
train4.head()


# In[196]:


train4.isnull().sum()


# In[197]:


train4.to_csv('train4.csv')


# In[1]:


park.head()


# In[ ]:




