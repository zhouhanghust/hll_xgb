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
from evaluation.CrossValidationKs import CrossValidationKs


def getCVResult(spark):
    train_df = Hdfs2Df.readHdfsCsv(spark=spark, data_path=config['TRAIN']['train_path'])
    test_df = Hdfs2Df.readHdfsCsv(spark=spark, data_path=config['TRAIN']['test_path'])

    missing = config['XGBOOST']['missing']
    train_df = PreProcessor.transColType(train_df, missing)
    test_df = PreProcessor.transColType(test_df, missing)
    train, train_col = PreProcessor.transVector(train_df, 'features')
    test, test_col = PreProcessor.transVector(train_df,'features')

    xgb_cv = CrossValidationKs(spark,test,config['TRAIN']['cv_num'])
    xgb_cv.crossValidationByXGB(config['XGBOOST'])


if __name__ == "__main__":
    spark = SparkEnv.getSession()
    getCVResult(spark)







