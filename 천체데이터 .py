#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm 
from time import time
import datetime
import gc


# In[2]:


from scipy.signal import find_peaks, peak_widths, peak_prominences


# In[3]:


from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


# In[4]:


from scipy.optimize import minimize
from sklearn.metrics import log_loss


# 베이지안 opt로 튜닝

# In[5]:


train = pd.read_csv('C:/Users/user/Desktop/2020취업준비/천체/train (2).csv')
test = pd.read_csv('C:/Users/user/Desktop/2020취업준비/천체/test (2).csv')


# In[6]:


train.shape


# In[7]:


test.shape


# In[8]:


train.head()


# In[9]:


train.info()


# In[10]:


train.isna().sum()


# In[11]:


train.describe()


# In[12]:


test.head()


# In[13]:


train['type'].unique()


# In[14]:


train['type'].nunique()


# In[15]:


train['type'].value_counts()


# In[16]:


plt.figure(figsize=(18,5))
sns.barplot(train['type'].value_counts().index, train['type'].value_counts().values)
plt.xticks(rotation=-45)


# imbalanced data: sampling, weight 적용가능

# # train, test 분포차이 확인하기

# In[29]:


for col in train.columns[2:]:
    df = train[[col]].loc[np.logical_and(train[col]<test[col].max(),
                                        train[col]>test[col].min())]
    
    fig, ((ax1, ax2)) = plt.subplots(1,2,figsize= (15,3))
    sns.distplot(train[col], label='train', ax=ax1)
    sns.distplot(test[col], label='test',  ax=ax1)
    sns.distplot(df[col], label='train', ax=ax2)
    sns.distplot(test[col], label='test', ax=ax2)
    if col == 'fiberID':
        fig.suptitle('Handing outliers:Before vs After', fontsize=18)
        
    ax1.set_title(col)
    ax1.legend()
    ax2.legend()
    plt.tight_layout()


# In[26]:


train.shape[0]


# In[17]:


train_shape = 199991

for col in train.columns[3:]:
  train = train.loc[np.logical_and(train[col]>test[col].min(), 
                                   train[col]<test[col].max())]

print('제거된 행 개수 :', 199991 - train.shape[0])


# 이상치 제거 후 eda 각 변수와 타겟변수의 관계

# In[18]:


for col in train.columns[2:]:
    plt.figure(figsize=(25,4))
    sns.boxplot(x='type',y=col, data=train)
    plt.title(col)
    plt.xticks(rotation=-30)
    plt.show()


# # fiberID빈도와 type 의 연관이 있는가

# In[19]:


train2 = train.copy()
test2 = test.copy()


# In[20]:


train2['t']=1
all_df = pd.concat([train2.drop('type', axis=1), test2], axis=0)
all_df


# In[21]:


all_df['count'] = all_df['fiberID'].map(all_df['fiberID'].value_counts())
all_df.head()


# In[22]:


all_df['fiberID'].value_counts()


# In[23]:


train2['t'] = 1
test2['t'] = 0
types = train2['type']
all_df = pd.concat([train2.drop('type', axis=1), test2], axis=0)

all_df['count'] = all_df['fiberID'].map(all_df['fiberID'].value_counts())
train2 = all_df[all_df['t']==1]
train2['type'] = types.values

train2 = train2.drop(columns='t')
test2 = test2.drop(columns='t')


# In[24]:


all_df.shape


# In[25]:


test2.shape


# In[26]:


train2.shape


# In[27]:


train2.head(15)


# In[28]:


sns.lmplot(x='fiberID', y='count', hue='type', data=train2, fit_reg=False, aspect=1.8, height=7,
          palette = sns.color_palette())
plt.show()


# # type 별 포인트 그래프 1개만 예시

# 아래거는 각 타입(QSO, 백색왜성 등)에서 
# psf_Mag_U의 값(a.columns[2])의 분포 라는 뜻

# In[49]:


for t in train2['type'].unique():
    a=train2[train2['type']==t]
    plt.figure(figsize=(20,5))
    plt.plot(a[a.columns[2]].T)
    plt.title(t)
    plt.show()


# In[29]:


a= train2[train2['type']=='QSO']
a


# In[30]:


train2['type']=='QSO'


# In[31]:


a.columns[2]


# In[32]:


a[a.columns[2:22]].T


# type별로 구분되는 그래프 형태를 확인할 수 있었음

# # prominence와 width

# In[54]:


b= np.linspace(0,6*np.pi,1000)
b= np.sin(b)+0.6*np.sin(2.6*b)


# In[56]:


peaks,_ =find_peaks(b)


# In[57]:


peaks


# In[59]:


prominences=peak_prominences(b, peaks)[0]
prominences


# In[60]:


contour_heights=b[peaks]-prominences


# In[67]:


contour_heights = b[peaks] - prominences
plt.plot(b)
plt.plot(peaks, b[peaks], "x")
plt.vlines(x=peaks, ymin=contour_heights, ymax=b[peaks])
plt.show()


# In[68]:


##prominences는 검은색 선의 거리!!!!


# In[72]:


results_half=peak_widths(b, peaks, rel_height=0.5)
results_half


# In[78]:


*results_half[1:]


# In[76]:


plt.plot(b)
plt.plot(peaks, b[peaks], "x")
plt.hlines(*results_half[1:], color="C2")
plt.show()


# # 가능한 변수:
# 앞뒤 포인트 차이(5개 포인트 간격까지) /
# 20개 포인트 랭킹 /
# peak의 폭과 너비 /

# In[79]:


a=train[train['type']=='STAR_PN']
x=a[a.columns[3:23]].iloc[3] #3행이라는 뜻
peaks,_ =find_peaks(x)
peaks


# In[80]:


a=train[train['type']=='STAR_PN']
a


# In[81]:


x


# In[82]:


x[peaks]


# In[83]:


prominences = peak_prominences(x, peaks)
prominences


# In[84]:


results_06=peak_widths(x, peaks, rel_height=0.6)
results_06


# In[85]:


results_03 = peak_widths(x, peaks, rel_height=0.3)
results_03


# In[86]:


results_03


# In[87]:


plt.plot(x)
plt.hlines(*results_03[1:], color="C3")


# In[88]:


a = train[train['type']=='STAR_PN']
x = a[a.columns[3:23]].iloc[3]
peaks, _ = find_peaks(x)
results_06 = peak_widths(x, peaks, rel_height=0.6)
results_03 = peak_widths(x, peaks, rel_height=0.3)
prominences = peak_prominences(x, peaks)[0]
contour_heights = x[peaks] - prominences

plt.figure(figsize=(10,4))
plt.plot(x)
plt.plot(peaks, x[peaks], "x")
plt.hlines(*results_03[1:], color="C3")
plt.hlines(*results_06[1:], color="C2")
plt.vlines(x=peaks, ymin=contour_heights, ymax=x[peaks])
plt.title('peak example')
plt.xticks(rotation=-45)
plt.show()


# In[89]:


def peak_width_mean(x, height):
    peaks,_ =find_peaks(x)
    results_full=peak_widths(x, peaks, rel_height=height)[0]
    
    return np.mean(results_full)

def peak_prominence_mean(x):
    peaks,_ =find_peaks(x)
    prominences = peak_prominences(x, peaks)[0]
    
    return np.mean(prominences)


# In[90]:


train2['num_peak'] = train2[train2.columns[2:22]].apply(lambda x : find_peaks(x)[0].shape[0], axis=1)
#데이터 행 마다의  peak 개수를 구하는 함수


# In[91]:


train2['peak_width_mean_3'] = train2[train2.columns[2:22]].apply(lambda x : peak_width_mean(x, 0.3), axis=1)


# In[92]:


train2['peak_width_mean_5'] = train2[train2.columns[2:22]].apply(lambda x : peak_width_mean(x, 0.5), axis=1)
train2['peak_width_mean_6'] = train2[train2.columns[2:22]].apply(lambda x : peak_width_mean(x, 0.6), axis=1)
train2['peak_prominence_mean'] = train2[train2.columns[2:22]].apply(lambda x : peak_prominence_mean(x), axis=1)


# In[94]:


train2.head(15)


# 
# # 변수생성

# In[96]:


train2=train.copy()
test2=test.copy()


# 

# In[97]:


for j in range(0,19):
        train2['diff_'+ str(j)] = train2[train2.columns[j+4]] - train2[train2.columns[j+3]]
        test2['diff_'+ str(j)] = test2[test2.columns[j+3]] - test2[test2.columns[j+2]]
        print(train2.columns[j+4], ' - ',train2.columns[j+3], j)


# In[98]:


train2.head()


# 20포인트 랭킹??

# In[99]:


mag_rank_tr= train2[train2.columns[3:23]].rank(axis=1)
mag_rank_tt = test2[test2.columns[2:22]].rank(axis=1)

rank_col=[]
for col in train2[train2.columns[3:23]].columns:
    col = col + '_rank'
    rank_col.append(col)
mag_rank_tr.columns= rank_col
mag_rank_tt.columns= rank_col

train2 = pd.concat([train2, mag_rank_tr], axis=1)
test2 = pd.concat([test2, mag_rank_tt], axis=1)


# In[100]:


pd.set_option('display.max_columns',100)


# In[101]:


train2.head(20)


# In[102]:


diff_col=[]
for col in ['u', 'g','r','i','z']:
    for i in range(3):
        diff_col.append(col + '_'+ str(i))
diff_col


# In[103]:


diff_col=[]
for col in ['u', 'g','r','i','z']:
    for i in range(3):
        diff_col.append(col + '_'+ str(i))
mag_wave_diff_tr=pd.DataFrame(np.zeros((train2.shape[0], 15)), index=train2.index) #train2.shape[0] 행을 나타냄
mag_wave_diff_tt=pd.DataFrame(np.zeros((testhape.shape[0]. 15)))

for i in range(0.15.5):
    for j in range(5):
        mag_wave_diff_tr.loc[:j+1]=train2[train2.columns[3+j]]-train2


# In[104]:


# 측정방법별 파장 차이 비교 변수
diff_col = []
for col in ['u','g','r','i','z']:
    for i in range(3):
        diff_col.append(col + '_' + str(i))
mag_wave_diff_tr = pd.DataFrame(np.zeros((train2.shape[0], 15)), index=train2.index)
mag_wave_diff_tt = pd.DataFrame(np.zeros((test2.shape[0], 15)))

for i in range(0,15,5):
    for j in range(5):
        mag_wave_diff_tr.loc[:,j+i] = train2[train2.columns[3+j]] - train2[train2.columns[8+j+i]]
        mag_wave_diff_tt.loc[:,j+i] = test2[test2.columns[2+j]] - test2[test2.columns[7+j+i]]
        print(test.columns[2+j], ' - ',test2.columns[7+j+i], i+j)


# In[105]:


mag_wave_diff_tr.columns = diff_col
mag_wave_diff_tt.columns = diff_col

train2 = pd.concat([train2, mag_wave_diff_tr], axis=1)
test2 = pd.concat([test2, mag_wave_diff_tt], axis=1)


# In[106]:


train2.head()


# # peak 관련 변수

# In[107]:


def peak_width_mean(x, height):
    peaks, _ = find_peaks(x)
    results_full = peak_widths(x, peaks, rel_height=height)[0]

    return np.mean(results_full)

def peak_prominence_mean(x):
    peaks, _ = find_peaks(x)
    prominences = peak_prominences(x, peaks)[0]

    return np.mean(prominences)


# In[108]:


train2['num_peak'] = train2[train2.columns[3:23]].apply(lambda x : find_peaks(x)[0].shape[0], axis=1)
train2['peak_width_mean_3'] = train2[train2.columns[3:23]].apply(lambda x : peak_width_mean(x, 0.3), axis=1)
train2['peak_width_mean_5'] = train2[train2.columns[3:23]].apply(lambda x : peak_width_mean(x, 0.5), axis=1)
train2['peak_width_mean_6'] = train2[train2.columns[3:23]].apply(lambda x : peak_width_mean(x, 0.6), axis=1)
train2['peak_prominence_mean'] = train2[train2.columns[3:23]].apply(lambda x : peak_prominence_mean(x), axis=1)

test2['num_peak'] = test2[test2.columns[2:22]].apply(lambda x : find_peaks(x)[0].shape[0], axis=1)
test2['peak_width_mean_3'] = test2[test2.columns[2:22]].apply(lambda x : peak_width_mean(x, 0.3), axis=1)
test2['peak_width_mean_5'] = test2[test2.columns[2:22]].apply(lambda x : peak_width_mean(x, 0.5), axis=1)
test2['peak_width_mean_6'] = test2[test2.columns[2:22]].apply(lambda x : peak_width_mean(x, 0.6), axis=1)
test2['peak_prominence_mean'] = test2[test2.columns[2:22]].apply(lambda x : peak_prominence_mean(x), axis=1)


# # fiberID value count 변수

# In[109]:


train2['t']=1
test2['t']=0
types =train2['type']
all_df=pd.concat([train2.drop('type', axis=1), test2], axis=0)
all_df['count']=all_df['fiberID'].map(all_df['fiberID'].value_counts())


# In[110]:


train2 = all_df[all_df['t']==1].drop('t', axis=1)
test2 = all_df[all_df['t']==0].drop('t', axis=1)

train2['type']= types.values


# In[111]:


tr_all_diff_0306 = train2.drop(columns=['peak_width_mean_5','id','type'])
tr_all_diff_05 = train2.drop(columns=['peak_width_mean_3', 'peak_width_mean_6','id','type'])

tt_all_diff_0306 = test2.drop(columns=['peak_width_mean_5','id'])
tt_all_diff_05 = test2.drop(columns=['peak_width_mean_3', 'peak_width_mean_6','id'])


# In[114]:


train2.head(20)


# In[115]:


tr_all_diff_0306.head()


# # 모델학습

# In[116]:


encoder= LabelEncoder()
y=encoder.fit_transform(train2['type'])


# In[119]:


y


# # ***kfold 연습

# In[136]:


from sklearn.model_selection import KFold


# In[140]:


X=np.arange(16).reshape((8,-1))
X


# In[141]:


y=np.arange(8).reshape((-1,1))
y


# In[138]:


X=np.arange(16).reshape((8,-1))
y=np.arange(8).reshape((-1,1))
kf=KFold(n_splits=4)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # xgboost

# In[124]:


n_splits =5
mlogloss=[]

xgb_oof_ver1 = np.zeros((train2.shape[0], 19))
xgb_pred_ver1 = np.zeros((test.shape[0], 19))


# In[ ]:


for data, X_test in [(tr_all_diff_0306, tt_all_diff_0306), 
                     (tr_all_diff_05, tt_all_diff_05)]:
    data=data.reset_index(drop=True)
    for seed in [100,300]:
        kfold=StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
        for fold, (trn_idx, val_idx) in enumerate(kfold.split(data, y)):
            X_train2, X_valid = data.loc[trn_idx], data.loc[val_idx]
            y_train2, y_valid= y[trn_idx], y[val_idx]
            
            dtrain=xgb.DMatrix(X_train2, label=y_train2)
            dvalid=xgb.DMatrix(X_valid, label=y_valid)
            watchlist= [(dtrain, 'train'), (dvalid, 'valid')]
            
            start_time =time()
            
            model=xgb.train(xgb_param, dtrain, 5000, evals=watchlist, early_stopping_round=50, verbose_eval=5000)
            mlogloss.append(model.best_score)
            
            #predict
            dtest=xgb.DMatrix(X_test)
            if data.shape[1]<130:
                xgb_pred_

