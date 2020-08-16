# -*- coding: UTF-8 -*-



class Hdfs2Df():
    key = ['order_id', 'ab_order', 'driver_id', 'user_id','dt','label']


    @classmethod
    def readHdfsCsv(cls, spark: 'sparkSession', data_path: 'string')-> 'sparkdf':
        df = spark.read.format("csv")\
            .option("inferSchema","true")\
            .option("header","true") \
            .option("delimiter","\t") \
            .load(data_path)
        return df


    @classmethod
    def readHive(cls, spark, table_name):
        df = spark.sql("select * from {0}".format(table_name))
        return df


    @classmethod
    def writeHdfsCsv(cls, df, hdfsPath, num_partition=200):
        df \
            .repartition(num_partition) \  # 同样这里可以指定分区字段
            .write \
            .format("csv") \
            .option("delimiter", "\t") \
            .option("header","true") \
            .mode("overwrite") \
            .csv(hdfsPath)



