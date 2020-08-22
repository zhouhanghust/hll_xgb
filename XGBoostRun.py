# encoding=utf-8

import numpy as np
import pandas as pd
import json, time, datetime
import os, sys
from utils.Hdfs2Df import Hdfs2Df


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
from params.params import config
from preProcessor.PreProcessor import PreProcessor
from utils.SavaTools import SavaTools
from classification.XGBoostClassifier import XGBoostClassifier
from evaluation.Evaluator import Evaluator



def train(spark):
    if config['XGBOOST']['checkpointInitialization'] == 'true':
        checkpoint_path = config['XGBOOST']['checkpoint_path']
        op = os.system("hadoop fs -rmr %s/*"%checkpoint_path)
        if not op:
            print("initialize checkpoint successfully.")
    train_df = Hdfs2Df.readHdfsCsv(spark=spark, data_path=config['TRAIN']['train_path'])
    test_df = Hdfs2Df.readHdfsCsv(spark=spark, data_path=config['TRAIN']['test_path'])

    missing = config['XGBOOST']['missing']
    train_df = PreProcessor.transColType(train_df, missing)
    test_df = PreProcessor.transColType(test_df, missing)
    train, train_col = PreProcessor.transVector(train_df, 'features')
    test, test_col = PreProcessor.transVector(train_df,'features')

    SavaTools.saveModelFeature(train_col, config['TRAIN']['local_model_feature_path'])
    xgb_handle = XGBoostClassifier(config['XGBOOST'])
    xgbModel = xgb_handle.trainAndSave(spark,train,config['TRAIN']['hdfs_model_path'])

    train_res, train_auc = xgb_handle.predict(spark, train, xgbModel)
    test_res, test_auc = xgb_handle.predict(spark,test, xgbModel)
    train_res.cache()
    test_res.cache()

    evaluator_handle = Evaluator(spark)
    train_ks = evaluator_handle.evaluateKs(train_res, 'train_res', 'score')
    train_auc = evaluator_handle.evaluateAuc(train_res,"score")
    test_ks = evaluator_handle.evaluateKs(test_res, 'test_ks', 'score')
    test_auc = evaluator_handle.evaluateAuc(test_res,"score")


    fscore = xgbModel.booster.getFeatureScore()
    xgb_handle.saveFeatureImportance(train_col,fscore,config['TRAIN']['local_model_feature_weights_path'],train_auc,test_auc,train_ks,test_ks)
    SavaTools.saveHdfsFile(train_res,config['TRAIN']['train_res_path'])
    SavaTools.saveHdfsFile(train_res, config['TRAIN']['test_res_path'])


if __name__ == "__main__":
    spark = SparkEnv.getSession()
    train(spark)


