# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 14:15:10 2018

@author: Administrator
"""
import datetime
import pandas as pd
from numpy import array
from sklearn.model_selection import train_test_split
#预测类目和属性的分割
#类目：属性1，属性2；类目2：rrrrr
def cat_pro_split(cp):
    lc = []
    lp = []
    lcp = cp.split(';')
    for c in lcp:
        cpsplit = c.split(':')
        lc.append(cpsplit[0])
        if len(cpsplit)>1:
            lp.extend(cpsplit[1].split(','))
    return pd.Series([lc,lp])

#广告类目和属性的分割
def item_split(ic):
    icsp = ic.split(';')
    return icsp

#统计广告和预测的类目属性个数，并计算预测正确比例
def count_ratio(df_ip,df_ic,df_pcp):
    icc = [] #item  category count  广告的类目的统计
    ipc = [] #广告的属性统计
    pcc = [] #预测的类目统计
    ppc = [] #预测的属性统计
    ipcc = []    #广告和预测的相同的类目的个数
    ippc = []    #广告和预测的相同的属性的个数
    print('开始类目和属性特征的计数\n')
    for i in range(len(df_ip)):
        icc.append(len(df_ic.iloc[i]))
        ipc.append(len(df_ip.iloc[i]))
        pcc.append(len(df_pcp.iloc[i,0]))
        ppc.append(len(df_pcp.iloc[i,1]))
        ic_set = set(df_ic.iloc[i])
        ip_set = set(df_ip.iloc[i])
        pc_set = set(df_pcp.iloc[i,0])
        pp_set = set(df_pcp.iloc[i,1])
        ipcc.append(len(ic_set & pc_set))
        ippc.append(len(ip_set & pp_set))
        if i in (10,100000,200000,300000,400000,478137):
            print('已完成统计%d%%' %int((i/478138)*100))
    return pd.DataFrame({'ipc_ra_ic':array(ipcc)/array(icc),'ipc_ra_ip':array(ipcc)/array(ipc),'ipp_ra_pc':array(ippc)/array(pcc),'ipp_ra_pp':array(ippc)/array(ppc)})
#计算广告商品营业额，某用户对一件商品/店铺/品牌的关注次数,用户浏览广告的平均价格
def id_featues(df):
    #新增商品营业额
    df['sumprice'] = df['item_price_level'] * df['item_sales_level']
    #shop_id为随机选取，计数无所谓
    df_uc = df.groupby('user_id')['shop_id'].count()
    df_uc.name = 'count_user'
    df_uc = df_uc.reset_index()
    df = pd.merge(df,df_uc,on='user_id')
    df_ap = df.groupby('user_id')['item_price_level'].mean().astype('int')
    df_ap.name = 'ave_price'
    df_ap = df_ap.reset_index()
    df = pd.merge(df,df_ap,on='user_id')
    id_list = ['item_id','shop_id','item_brand_id','item_city_id']
    for i in id_list:
        df_ci = df.groupby(['user_id',i])['count_user'].count()
        df_ci.name = i + '_c'
        df = pd.merge(df,df_ci.reset_index(),on=['user_id',i])
    return df
#时间处理函数
def hour_format(df):
    df_time = df['context_timestamp'].apply(datetime.datetime.fromtimestamp,1)
    #format_time = []
    df_time = df_time.apply(lambda x : pd.Series([x.hour,x.day]))
    df_time.columns = ['hours','days']
    '''
    可以把时间晚上七点到十点作为特殊时段
    for i in df_time:
        if i in (19,20,21,22):
            format_time.append(1)
        else:
            format_time.append(0)
    '''
    df.drop(['context_timestamp'],1,inplace=True)
    df = pd.concat([df,df_time],1)
    return df

#主函数        
def main(path):
    df = pd.read_csv(path,sep=' ')
    df1 = df[['instance_id','item_category_list','item_property_list','predict_category_property']]
    print('开始预测变量的分割\n')
    df_pcp = df1['predict_category_property'].apply(cat_pro_split)
    print('已完成预测类目和属性的分割,形状为：',df_pcp.shape)
    df_ip = df1['item_property_list'].apply(item_split)
    df_ic = df1['item_category_list'].apply(item_split)
    print('已完成广告类目和属性的分割,形状为：',df_ic.shape,df_ip.shape)
    df_count = count_ratio(df_ip,df_ic,df_pcp) 
    print('已完成广告类目和属性新增变量的计算\n')
    df_all = df.drop(['item_category_list','item_property_list','predict_category_property'],1)
    df_all = pd.concat([df_all,df_count],1)
    id_featues(df_all)
    print('已完成用户各项关注次数统计\n')
    df_all = hour_format(df_all)
    print('已完成时间转换\n')
    return df_all
if '__init__' == '__main__':
    path = '../data/round1_ijcai_18_train_20180301.txt'
    data = main(path)
    print(data.columns)
    #切分训练集和测试集分别保存，备用
    data.to_csv('../data/round1_train_countfeatures.csv',header=True,index=None)
   
