# encoding=utf-8
import numpy as np
import pandas as pd
import json, time, datetime
import os, sys

os.environ['PYSPARK_PYTHON'] = './py_env/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/data/nicolas.zhou/opt/python3/bin/python3'
os.environ['SPARK_HOME'] = '/usr/hdp/current/spark2-client'

# 添加pyspark包路径
sys.path.append('/usr/hdp/current/spark2-client/python')
sys.path.append('/usr/hdp/current/spark2-client/python/lib/py4j-0.10.7-src.zip')


from dataAnalyzer.utils.SparkEnv import SparkEnv



spark = SparkEnv.getSession()

tr_begin, tr_end = datetime.date(2020, 5, 3), datetime.date(2020, 7, 4)
te_begin, te_end = datetime.date(2020, 7, 5), datetime.date(2020, 7, 11)

df = spark.sql("select * from temp.zh_samples limit 2000")
print(df.show(10))

# 模型使用特征
cols_feature = [
    #### order & context
    "has_been_rejected_once",
    "create_hour",
    "order_hour",
    "is_appointment",
    "order_distance",

    "order_type",
    "is_prepaid",
    "order_vehicle_id",
    "pickup_dist",
    "is_with_remark",

    "path_point_num",
    "is_cross_city",
    "follow_person_num_from_remark",
    "order_vehicle_name_enum",
    "city_id",

    "end_city_id",
    "length_remark",
    "create_hour_minute",
    "order_hour_minute",
    "total_price_fen",

    "tips_price_fen",
    "weekday",
    "is_special_request_cart",
    "is_special_request_photo_receipt",
    "is_special_request_paper_receipt",

    "is_special_request_move",
    "is_special_request_return",
    "order_is_request_tailgate",
    "order_is_request_seat_type",

    #### driver
    "driver_age",
    "driver_vip_level_enum",
    "total_pickup_num",
    "driver_avg_completed_transation_per_order",
    "driver_complete_rate"
    ]

# 在模型特征的基础上，添加一些其他信息
cols_non_feature_info = ['order_id', 'ab_order', 'driver_id', 'user_id', 'dt', 'label']

# 新添加特征名称（待在模型中尝试)
cols_feature_more = []

cols_feature_all = cols_feature + cols_feature_more

# 数据类型全部转换为 double
col_strs1 = ["NVL(cast({0} as double), -1) {0}".format(c0) for c0 in cols_feature_all] + cols_non_feature_info
# col_strs2: 数据预处理,将所有0值转换为-0.1, 目的是解决spark sparseVector 处理缺失值机制带来的线上线下不一致的问题
col_strs2 = ["if({0} = 0, -0.1, {0}) {0}".format(c0) for c0 in cols_feature_all] + cols_non_feature_info
sdf = df.selectExpr(*col_strs1) \
    .selectExpr(*col_strs2) \
    .filter("order_vehicle_id <> -1") \
    .filter("order_type >= 1") \
    .filter("order_type <= 2").cache()



import datetime
from pyspark.sql.functions import col

# 筛选对应日期；排除6.17, 6.24脏数据
sel_rows_tr = (col("dt") >= tr_begin) & (col("dt") <= tr_end) & (col("dt") != "2020-06-17") & (col("dt") != "2020-06-24")
sel_rows_te = (col("dt") >= te_begin) & (col("dt") <= te_end) & (col("dt") != "2020-06-17") & (col("dt") != "2020-06-24")

sdf_tr = sdf.filter(sel_rows_tr).select(*[cols_feature_all + ["label"]]).cache()
sdf_te = sdf.filter(sel_rows_te).select(*[cols_feature_all + ["label"]]).cache()

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

va = VectorAssembler().setInputCols(cols_feature_all).setOutputCol('features')

dims_analysis = ['order_vehicle_id', 'create_hour', 'order_type']
sdf_tr1 = va.transform(sdf_tr).select(['features', 'label'] + dims_analysis) # Note: 加入order_vehicle_id维度便于统计分车型auc
sdf_te1 = va.transform(sdf_te).select(['features', 'label'] + dims_analysis)

from sparkxgb import XGBoostEstimator

xgb = XGBoostEstimator(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    objective='binary:logistic',
    eval_metric='auc',
    nworkers=64,
    nthread=4,
    num_round=200,
    eta=0.05,
    max_depth=8,
    min_child_weight=10.0,
    subsample=0.8,
    gamma=1.0
)

xgbmodel = xgb.fit(sdf_tr1)

print(xgbmodel.transform(sdf_te1).show(10))




