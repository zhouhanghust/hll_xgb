# encoding=utf-8
from pyspark.sql import SparkSession

class SparkEnv():
    _spark = SparkSession.builder \
        .appName('test') \
        .config('spark.yarn.queue', 'idphive') \
        .config('spark.dynamicAllocation.enabled', True) \
        .config('spark.shuffle.service.enabled', True) \
        .config('spark.executor.memory', '8g') \
        .config('spark.driver.memory', '128g') \
        .config('spark.dynamicAllocation.minExecutors', 16) \
        .config('spark.dynamicAllocation.maxExecutors', 64) \
        .config('spark.executor.cores', 4) \
        .config('spark.task.cpus', 4) \
        .config('spark.default.parallelism', 32) \
        .config('spark.sql.shuffle.partitions', 64) \
        .config('spark.hadoop.hive.exec.dynamic.partition', True) \
        .config('spark.hadoop.hive.exec.dynamic.partition.mode', 'nonstrict') \
        .config('spark.sql.sources.partitionOverwriteMode', 'dynamic') \
        .config('spark.yarn.dist.archives','hdfs:///user/yuan.shi/spark/py/py_env.tar.gz#py_env') \
        .config('spark.jars','hdfs:///user/yuan.shi/spark/jar/xgboost4j-0.72.jar,hdfs:///user/yuan.shi/spark/jar/xgboost4j-spark-0.72.jar') \
        .enableHiveSupport() \
        .getOrCreate()



    @classmethod
    def getSession(cls):
        cls._spark.sparkContext.addPyFile('hdfs:///user/yuan.shi/spark/py/sparkxgb.zip')
        return cls._spark





