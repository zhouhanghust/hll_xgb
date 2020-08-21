# encoding=utf-8
import numpy as np
import pandas as pd
import json, time, datetime
import os, sys

os.environ['PYSPARK_PYTHON'] = '/data/zhouhang/py_env/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/data/zhouhang/py_env/bin/python3'
os.environ['SPARK_HOME'] = '/usr/hdp/current/spark2-client'

# 添加pyspark包路径
sys.path.append('/usr/hdp/current/spark2-client/python')
sys.path.append('/usr/hdp/current/spark2-client/python/lib/py4j-0.10.7-src.zip')
# 添加sparkxgb包路径
sys.path.append('/data/zhouhang/sparkxgb_all')
sys.path.append('/data/zhouhang/sparkxgb_all/xgboost4j-spark-0.72.jar')
sys.path.append('/data/zhouhang/sparkxgb_all/xgboost4j-0.72.jar')


from dataAnalyzer.utils.SparkEnv import SparkEnv



spark = SparkEnv.getSession()

tr_begin, tr_end = datetime.date(2020, 5, 3), datetime.date(2020, 7, 4)
te_begin, te_end = datetime.date(2020, 7, 5), datetime.date(2020, 7, 11)

SQL0 = """
    create table temp.zh_samples
    as
    select 
        a.order_id, a.driver_id, a.label, a.ab_order, a.user_id, a.dt,
        has_been_rejected_once, create_hour, order_hour, is_appointment, order_distance, order_type, is_prepaid, order_vehicle_id, pickup_dist, is_with_remark, path_point_num, is_cross_city, follow_person_num_from_remark, order_vehicle_name_enum, city_id, end_city_id, length_remark, create_hour_minute, order_hour_minute, total_price_fen, tips_price_fen, weekday, is_special_request_cart, is_special_request_photo_receipt, is_special_request_paper_receipt, is_special_request_move, is_special_request_return, order_is_request_tailgate, order_is_request_seat_type, driver_age, driver_vip_level_enum, total_pickup_num, driver_avg_completed_transation_per_order, driver_complete_rate, be_favorite_rate, define_vehicle_type_enum, vehicle_type_enum, driver_appointment_order_canceled_rate_90d, driver_prepaid_order_canceled_rate_90d, driver_cash_order_canceled_rate_90d, driver_canceled_order_3q_pickup_duration_90d, driver_completed_order_1q_pickup_distance_90d, driver_immediate_order_canceled_rate_30d, driver_prepaid_order_canceled_rate_30d, driver_completed_order_avg_pickup_distance_30d, driver_completed_order_2q_order_distance_14d, driver_appointment_canceled_rate_180d, driver_cash_canceled_rate_180d, driver_is_brokerage_canceled_rate_180d, vehicle_transport_type, vehicle_power_type, drv_respond_sum, drv_veh_cancel_rate_0, drv_veh_cancel_rate_1, drv_veh_cancel_rate_2, drv_veh_cancel_rate_3, drv_veh_cancel_rate_4, drv_veh_cancel_rate_5, drv_veh_cancel_rate_6, drv_veh_cancel_rate_7, drv_veh_cancel_rate_8, drv_veh_cancel_rate_9, drv_veh_cancel_rate_10, drv_veh_cancel_rate_11, drv_veh_rate_0, drv_veh_rate_1, drv_veh_rate_2, drv_veh_rate_3, drv_veh_rate_4, drv_veh_rate_5, drv_veh_rate_6, drv_veh_rate_7, drv_veh_rate_8, drv_veh_rate_9, drv_veh_rate_10, drv_veh_rate_11, driver_90d_reject_reason1, driver_90d_reject_reason2, driver_90d_special_request_c, driver_90d_special_request_d, driver_90d_special_request_e, driver_90d_special_request_f, driver_90d_special_request_g, driver_start_city_id, order_pk_start_city_driver_major_city_order_rate, driver_end_city_id, order_pk_end_city_driver_major_city_order_rate, vehicle_is_with_tailgate, vehicle_seat_type_enum, vehicle_height_limitation_metre, vehicle_compartment_type_enum, vehicle_roof_type_enum, vehicle_is_steel_car, member_no, user_created_order_num2, user_completed_rate2, user_prepaid_order_canceled_rate_90d, user_cash_order_canceled_rate_90d, user_completed_order_1q_pickup_distance_90d, user_completed_order_2q_order_distance_90d, user_cash_canceled_rate_180d, user_is_brokerage_canceled_rate_180d, ib_order_veh_sum, user_order_sv_ratio, user_order_mv_ratio, user_order_sc_ratio, user_order_mc_ratio, user_order_lc_ratio, usr_veh_cancel_rate_0, usr_veh_cancel_rate_1, usr_veh_cancel_rate_2, usr_veh_cancel_rate_3, usr_veh_cancel_rate_4, usr_veh_cancel_rate_5, usr_veh_cancel_rate_6, usr_veh_cancel_rate_7, usr_veh_cancel_rate_8, usr_veh_cancel_rate_9, usr_veh_cancel_rate_10, usr_veh_cancel_rate_11, usr_veh_rate_0, usr_veh_rate_1, usr_veh_rate_2, usr_veh_rate_3, usr_veh_rate_4, usr_veh_rate_5, usr_veh_rate_6, usr_veh_rate_7, usr_veh_rate_8, usr_veh_rate_9, usr_veh_rate_10, usr_veh_rate_11, user_90d_reject_reason1, user_90d_reject_reason2, user_90d_special_request_c, user_90d_special_request_d, user_90d_special_request_e, user_90d_special_request_f, user_90d_special_request_g

    from (
        select * from algorithm.fitness_m11_feature_order 
        where dt between "{0}" and "{1}" and driver_id <> 0
    ) a
    left join (
        select * from algorithm.fitness_m11_feature_driver 
        where dt between date_sub("{0}", {2}) and date_sub("{1}", {2}) and driver_id <> 0
    ) b
        on a.driver_id = b.driver_id and date_sub(a.dt, {2}) = b.dt
    left join (
        select * from algorithm.fitness_m11_feature_user 
        where dt between date_sub("{0}", {2}) and date_sub("{1}", {2})
    ) c    
        on a.user_id = c.user_id and date_sub(a.dt, {2}) = c.dt

""".format(str(tr_begin), str(te_end), 1)

# 将样本+特征存为一张表供后续使用 temp.zh_samples
# spark.sql(SQL0)
# df = spark.sql(SQL0)
# print(df.show(10))

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
    "driver_complete_rate",

    "be_favorite_rate",
    "define_vehicle_type_enum",
    "vehicle_type_enum",
    "driver_appointment_order_canceled_rate_90d",
    "driver_prepaid_order_canceled_rate_90d",

    "driver_cash_order_canceled_rate_90d",
    "driver_canceled_order_3q_pickup_duration_90d",
    "driver_completed_order_1q_pickup_distance_90d",
    "driver_immediate_order_canceled_rate_30d",
    "driver_prepaid_order_canceled_rate_30d",

    "driver_completed_order_avg_pickup_distance_30d",
    "driver_completed_order_2q_order_distance_14d",
    "driver_appointment_canceled_rate_180d",
    "driver_cash_canceled_rate_180d",
    "driver_is_brokerage_canceled_rate_180d",

    "vehicle_transport_type",
    "vehicle_power_type",
    "drv_respond_sum",
    "drv_veh_cancel_rate_0",  # 不同车型的历史取消率
    "drv_veh_cancel_rate_1",

    "drv_veh_cancel_rate_2",
    "drv_veh_cancel_rate_3",
    "drv_veh_cancel_rate_4",
    "drv_veh_cancel_rate_5",
    "drv_veh_cancel_rate_6",

    "drv_veh_cancel_rate_7",
    "drv_veh_cancel_rate_8",
    "drv_veh_cancel_rate_9",
    "drv_veh_cancel_rate_10",
    "drv_veh_cancel_rate_11",

    "drv_veh_rate_0",  # 不同车型的订单量占比
    "drv_veh_rate_1",
    "drv_veh_rate_2",
    "drv_veh_rate_3",
    "drv_veh_rate_4",

    "drv_veh_rate_5",
    "drv_veh_rate_6",
    "drv_veh_rate_7",
    "drv_veh_rate_8",
    "drv_veh_rate_9",

    "drv_veh_rate_10",
    "drv_veh_rate_11",
    "driver_90d_reject_reason1",  # from xuewei table
    "driver_90d_reject_reason2",
    "driver_90d_special_request_c",  # cart

    "driver_90d_special_request_d",  # photo
    "driver_90d_special_request_e",  # paper
    "driver_90d_special_request_f",  # move
    "driver_90d_special_request_g",  # return
    "driver_start_city_id",

    "order_pk_start_city_driver_major_city_order_rate",
    "driver_end_city_id",
    "order_pk_end_city_driver_major_city_order_rate",
    "vehicle_is_with_tailgate",  # 线上传司机画像即可 vehicle_ab_base
    "vehicle_seat_type_enum",

    "vehicle_height_limitation_metre",
    "vehicle_compartment_type_enum",
    "vehicle_roof_type_enum",
    "vehicle_is_steel_car",

    #### 10.3 user
    "member_no",
    "user_created_order_num2",
    "user_completed_rate2",
    "user_prepaid_order_canceled_rate_90d",
    "user_cash_order_canceled_rate_90d",

    "user_completed_order_1q_pickup_distance_90d",
    "user_completed_order_2q_order_distance_90d",
    "user_cash_canceled_rate_180d",
    "user_is_brokerage_canceled_rate_180d",
    "ib_order_veh_sum",

    "user_order_sv_ratio",
    "user_order_mv_ratio",
    "user_order_sc_ratio",
    "user_order_mc_ratio",
    "user_order_lc_ratio",

    "usr_veh_cancel_rate_0",
    "usr_veh_cancel_rate_1",
    "usr_veh_cancel_rate_2",
    "usr_veh_cancel_rate_3",
    "usr_veh_cancel_rate_4",

    "usr_veh_cancel_rate_5",
    "usr_veh_cancel_rate_6",
    "usr_veh_cancel_rate_7",
    "usr_veh_cancel_rate_8",
    "usr_veh_cancel_rate_9",

    "usr_veh_cancel_rate_10",
    "usr_veh_cancel_rate_11",
    "usr_veh_rate_0",
    "usr_veh_rate_1",
    "usr_veh_rate_2",

    "usr_veh_rate_3",
    "usr_veh_rate_4",
    "usr_veh_rate_5",
    "usr_veh_rate_6",
    "usr_veh_rate_7",

    "usr_veh_rate_8",
    "usr_veh_rate_9",
    "usr_veh_rate_10",
    "usr_veh_rate_11",
    "user_90d_reject_reason1",

    "user_90d_reject_reason2",
    "user_90d_special_request_c",  # cart
    "user_90d_special_request_d",  # photo
    "user_90d_special_request_e",  # paper
    "user_90d_special_request_f",  # move

    "user_90d_special_request_g",  # return
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

print(sdf.show(10))

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
    num_round=1000,
    eta=0.05,
    max_depth=8,
    min_child_weight=10.0,
    subsample=0.8,
    gamma=1.0
)

xgbmodel = xgb.fit(sdf_tr1)

xgbmodel.booster.saveModel("./xgb_model")




