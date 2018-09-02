# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:33:17 2018

@author: Administrator
"""
from sklearn.metrics import log_loss
import time
import pandas as pd 
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

#导入数据
path = '../data/round1_train_countfeatures.csv'
df = pd.read_csv(path)
df = df.fillna(0)
print(df['days'].value_counts())
df_train = df[(df['days']>=18) & (df['days']<=23)]
df_test = df[df['days']==24]
x_train = df_train.drop(['is_trade'],1)
x_test = df_test.drop(['is_trade'],1)
y_train = df_train['is_trade']
y_test = df_test['is_trade']

#用gbdt建模
time1_0 = time.time()
gbdt = GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,max_depth=3,
                                  min_samples_split=2,min_samples_leaf=1)
gbdt.fit(x_train,y_train)
test_gb_prob = gbdt.predict_proba(x_test)
train_gb_prob = gbdt.predict_proba(x_train)
print('gbdt的训练集log损失：',log_loss(y_train,train_gb_prob))
print('gbdt的测试集log损失：',log_loss(y_test,test_gb_prob))
time1_1 = time.time()
print('gbdt计算时间',time1_1 - time1_0)

#用xgboost建模
time2_0 = time.time()
xgb = XGBClassifier(max_depth=3,learning_rate=0.1,n_estimators=100,silent=False,
                    objective="binary:logistic",reg_alpha=0,reg_lambda=1)
xgb.fit(x_train,y_train)
test_xg_prob = xgb.predict_proba(x_test)
train_xg_prob = xgb.predict_proba(x_train)
print('xgboost的训练集log损失',log_loss(y_train,train_xg_prob))
print('xgboost的测试集log损失',log_loss(y_test,test_xg_prob))
time2_1 = time.time()
print('xgboost计算时间',time2_1 - time2_0)
#用lgb建模
time3_0 = time.time()
lgb = LGBMClassifier(objective='binary',learning_rate=0.02,
        n_estimators=100,
        num_leaves=45,
        depth=12,
        colsample_bytree=0.8,
        min_child_samples=14,
        subsample=0.9)
lgb.fit(x_train,y_train)
test_lgb_prob = lgb.predict_proba(x_test)
train_lgb_prob = lgb.predict_proba(x_train)
print('lightgbm的训练集log损失',log_loss(y_train,train_lgb_prob))
print('lightgbm的测试集集log损失',log_loss(y_test,test_lgb_prob))
time3_1 = time.time()
print('lightgbm计算时间',time3_1 - time3_0)
'''
#验证集输出结果，线上测试
import getFearures01
path_test = '../data/round1_ijcai_18_test_b_20180418.txt'
test_df = getFearures01.cpfeature(path_test)
test_pre = lgb.predict_proba(test_df)
result = pd.DataFrame({'instance_id':test_df['instance_id'],'predicted_score':test_pre[:,1]})
result.to_csv('./result.csv',sep=' ',header=True,index=None)'''