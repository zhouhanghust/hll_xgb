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


