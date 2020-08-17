# -*- coding: UTF-8 -*-
from hll_xgb.dataAnalyzer.run_ks_psi import Run_ks_psi
from pyspark.sql.functions import udf,col
from pyspark.sql.types import IntegerType
from pyspark.ml.evaluation import BinaryClassificationEvaluator

class Evaluator():
    def __init__(self, spark: 'SparkSession'):
        self._spark = spark


    def evaluateConfusionMatrix(self, df: 'sparkdf'):
        get_label = udf(lambda score: 1 if score >0.5 else 0, IntegerType())
        res = df.withColumn("label1", get_label(col("score")))
        # ex: [Row(label=0.0, count=8700328), Row(label=1.0, count=31467287)]
        classnum = res.groupBy("label").count().collect()
        pos_num = int(classnum[0][1])
        neg_num = int(classnum[1][1])
        pretrue = res.filter(col("label")==col("label1")).groupBy("label").count().collect()
        TP = int(pretrue[0][1])
        TN = int(pretrue[1][1])
        FP = neg_num - TN
        FN = pos_num - TP
        pos_precision = round(TP / (TP + FP),3)
        pos_recall = round(TP / (TP + FN),3)
        return TP, FP, TN, FN, pos_precision, pos_recall


    def evaluateAuc(self, df: 'sparkdf'):
        evaluator = BinaryClassificationEvaluator(rawPredictionCol="score", labelCol="label", metricName="areaUnderROC")
        auc = evaluator.evaluate(df)
        return auc

    # 用于计算特征的ks
    def evaluateKsCustomized(self, predictions: 'sparkdf', prob ="score"):
        ks_info = Run_ks_psi.getKsInfo(predictions, prob, 10, 3)
        ks_detail = ks_info['ks_detail']
        # todo 增加printKs功能


    # 计算模型分的ks
    def evaluateKs(self):
        pass

    def compute_ks(self):
        pass

    def get_src_quantiles(self):
        pass
    def percentile(self):
        pass







