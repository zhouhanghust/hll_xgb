# encoding=utf-8
from pyspark.sql import SparkSession

class SparkEnv():
    _spark = \
        SparkSession \
            .builder \
            .master("yarn") \
            .appName('spark_analysis') \
            .config('spark.yarn.queue', 'ai_intelligence') \
            .config('spark.driver.maxResultSize', '128g') \
            .config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer') \
            .config('spark.executor.memory', '16g') \
            .config('spark.driver.memory', '32g') \
            .config('spark.default.parallelism', 96) \
            .config('spark.sql.shuffle.partitions', 96) \
            .config('spark.executor.cores', 4) \
            .config('spark.dynamicAllocation.enabled', True) \
            .config('spark.shuffle.service.enabled', True) \
            .config('spark.dynamicAllocation.minExecutors', 16) \
            .config('spark.dynamicAllocation.maxExecutors', 16) \
            .config('spark.executor.instances', 16) \
            .config('spark.hadoop.hive.exec.dynamic.partition', True) \
            .config('spark.hadoop.hive.exec.dynamic.partition.mode', 'nonstrict') \
            .config('spark.sql.sources.partitionOverwriteMode', 'dynamic') \
            .config('spark.jars', 'hdfs:///user/yuan.shi/spark/jar/xgboost4j-0.90.jar,hdfs:///user/yuan.shi/spark/jar/xgboost4j-spark-0.90.jar') \
            .enableHiveSupport() \
            .getOrCreate()

    # .config('spark.yarn.dist.archives' ,'hdfs:///user/yuan.shi/spark/py/py_env.tar.gz#py_env') \
    # .config('spark.pyspark.driver.python', './py_env/bin/python') \
    # .config('spark.pyspark.python', './py_env/bin/python') \


    @classmethod
    def getSession(cls):
        cls._spark.sparkContext.addPyFile('hdfs:///user/yuan.shi/spark/py/sparkxgb.zip')
        return cls._spark





