# alimama
alimama IJCAI-18 阿里妈妈搜索广告转化预测
IJCAI-18 阿里妈妈搜索广告转化预测
https://tianchi.aliyun.com/competition/information.htm?spm=5176.11165320.5678.2.70196602l0Nh0e&raceId=231647

搜索广告的购买率预测 数据说明 本次比赛为参赛选手提供了5类数据（基础数据、广告商品信息、用户信息、上下文信息和店铺信息）。
 基础数据表提供了搜索广告最基本的信息，以及“是否交易”的标记。
 广告商品信息、用户信息、上下文信息和店铺信息等4类数据，提供了对转化率预估可能有帮助的辅助信息。

包含的信息包括，广告本身字段信息，客户个人信息，店铺的城市 、评分、收藏、被购买指标，用户的点击信息（时间、次数、 点击广告与搜索字段的匹配程度），
用户的品牌和店铺的忠诚指数

1、删除几个无关字段'instance_id','item_id','user_id' 
2、测试集是最后一天的数据，时间保存在days这个变量里 
3、目标分类变量是 is_trade 是否被交易 
4、使用决策树以及随机森林来调参、或者使用svm或logistic回归

getFeatures01 训练样本大小554.8MB，样本个数478138 开始预测变量的分割

已完成预测类目和属性的分割,形状为： (478138, 2) 已完成广告类目和属性的分割,形状为： (478138,) (478138,) 开始类目和属性特征的计数

已完成统计0% 已完成统计20% 已完成统计41% 已完成统计62% 已完成统计83% 已完成统计99%

已完成广告类目和属性新增变量的计算

已完成用户各项关注次数统计

已完成时间转换，30维特征

Index(['instance_id', 'item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_id', 'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level', 'context_id', 'context_page_id', 'shop_id', 'shop_review_num_level', 'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery', 'shop_score_description', 'is_trade', 'ipc_ra_ic', 'ipc_ra_ip', 'ipp_ra_pc', 'ipp_ra_pp', 'sumprice', 'hours', 'days'], dtype='object')

modeltrain02----------------------------------------------------------------
gbdt的训练集log损失： 0.08935791833109166
gbdt的测试集log损失： 0.08265368198396883
gbdt计算时间 108.50479984283447

xgboost的训练集log损失 0.09004494429642025
xgboost的测试集log损失 0.0826791516813898
xgboost计算时间 61.30361080169678

lightgbm的训练集log损失 0.08802450302572135
lightgbm的测试集集log损失 0.08282664986274751
lightgbm计算时间 17.9807391166687
