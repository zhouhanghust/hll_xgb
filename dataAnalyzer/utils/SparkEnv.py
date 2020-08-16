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
            .enableHiveSupport() \
            .getOrCreate()

    @classmethod
    def getSession(cls):
        return cls._spark





