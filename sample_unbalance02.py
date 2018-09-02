# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 12:25:18 2018

@author: Administrator
"""
import pandas as pd
import numpy as np

#from sklearn.ensemble import GradientBoostingClassifier
#数据处理并分割

#下采样
def subsample(df,number=15000):
    un_index = df[df['is_trade']==0].index
    no_index = np.array(df[df['is_trade']==1].index)
    random_un_index = np.random.choice(un_index,number,replace=False)
    ramdom_un_index = np.array(random_un_index)
    under_sample = np.concatenate([ramdom_un_index,no_index])
    df1 = df.iloc[under_sample,:]
    print(df1.shape)
    print(df1['is_trade'].value_counts())
    return df1
#过采样
def oversample(df,number=100000):
    un_index = np.array(df[df['is_trade']==0].index)
    no_index = df[df['is_trade']==1].index
    random_no_index = np.random.choice(no_index,number,replace=True)
    ramdom_no_index = np.array(random_no_index)
    over_sample = np.concatenate([ramdom_no_index,un_index])
    df2 = df.iloc[over_sample,:]
    print(df2.shape)
    print(df2['is_trade'].value_counts())
    return df2
#通过kmeans的中心点替代样本
def kmeansfeatuer(df,k=10000):
    from sklearn.cluster import MiniBatchKMeans
    df_name = list(df)
    df0 = df[df['is_trade']==0].reset_index(drop=True)
    df1 = df[df['is_trade']==1].reset_index(drop=True)
    km = MiniBatchKMeans(n_clusters=k,batch_size=80000)
    km.fit(df0)
    df0 = km.cluster_centers_
    df0 = pd.DataFrame(df0,columns=df_name)
    df = pd.concat([df0,df1]).reset_index(drop=True)
    return df
#通过单分类建模选择样本
def oneclassF(df):
    from sklearn.ensemble import IsolationForest
    ilf = IsolationForest(contamination=0.2)
    ilf.fit(df.drop(['is_trade'],1))
    errorlist = ilf.predict(df.drop(['is_trade'],1))
    df['error'] = errorlist
    return df[(df['error']==-1)&(df['is_trade']==0)].index


