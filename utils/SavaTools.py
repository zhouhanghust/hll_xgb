# -*- coding: UTF-8 -*-


class SavaTools():
    @staticmethod
    def saveModelFeature(traincols: 'list', local_model_feature_path: 'String')-> 'None':
        with open(local_model_feature_path,"w") as f:
            for col in traincols:
                f.write(col+"\n")

    @staticmethod
    def saveHdfsFile(df: 'sparkdf', res_path: 'String', num_partition=200):
        df \
            .repartition(num_partition) \ # 这里还可以指定列，.repartition(num_partition,"col1")
            .write \
            .format("csv") \
            .option("delimiter", "\t") \
            .option("header", "true") \
            .mode("overwrite") \
            .csv(res_path)


        df \
            .repartition(num_partition) \ # 这里还可以指定列，.repartition(num_partition,"col1")
            .write \
            .mode("overwrite") \
            .partitionBy('col1') \ # 按照分区来存
            .format("csv") \
            .option("delimiter", "\t") \
            .option("header", "true") \
            .csv(res_path)











