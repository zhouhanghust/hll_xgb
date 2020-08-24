# -*- coding: UTF-8 -*-
from dataAnalyzer.run_ks_psi import Run_ks_psi
from pyspark.sql.functions import udf,col
from pyspark.sql.types import IntegerType
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer
from params.params import whitelist,ks_detail_out_path, ks_summary_out_path


class Evaluator():
    def __init__(self, spark: 'SparkSession'):
        self._spark = spark
        self.whitelist = whitelist
        self.ks_detail_out_path = ks_detail_out_path
        self.ks_summary_out_path = ks_summary_out_path


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


    def evaluateAuc(self, df: 'sparkdf',rawPredictionCol="score", labelCol="label"):
        evaluator = BinaryClassificationEvaluator(rawPredictionCol=rawPredictionCol, labelCol=labelCol, metricName="areaUnderROC")
        auc = evaluator.evaluate(df)
        return auc

    # 用于计算特征的ks
    def evaluateKsCustomized(self, predictions: 'sparkdf', prob ="score", calc_specific=whitelist, ks_detail_out_path=ks_detail_out_path,ks_summary_out_path=ks_summary_out_path):
        Run_ks_psi.getKsInfo(predictions, prob, 10, 3, calc_specific,ks_detail_out_path,ks_summary_out_path)





    # 计算模型分的ks
    def evaluateKs(self, predictions: 'sparkdf', tableName:'string', prob:'string' = "core")->'double':
        predictions.createOrReplaceTempView(tableName)
        result = self.spark.sql("SELECT score as prob, label FROM %s"%tableName)
        viewName = tableName + "_result"
        result.createOrReplaceTemView(viewName)
        quantileDiscretizer = QuantileDiscretizer(numBuckets=10, inputCol='prob',outputCol='prob_cut')
        discreDF = quantileDiscretizer.fit(result).transform(result)
        cut_view_name = viewName + "_with_cut"
        discreDF.createOrReplaceTemView(cut_view_name)
        sql_str = r"SELECT count(label) as label_all, sum(label) as label_bad, min(prob) as min,  max(prob) as max, prob_cut FROM " + cut_view_name + " group by prob_cut order by prob_cut"
        resultLocal = spark.sql(sql_str).collect()
        ks, ks_cut_local = self.compute_ks(resultLocal)
        print(r"ks:\t%s"%str(ks))
        for line in ks_cut_local:
            print(line)

        return float(ks)




    def compute_ks(self, resultLocal: 'list[Row]')->'list[string]':
        result = list()
        cum_bad_arr = list()
        cum_good_arr = list()
        count = 0.0
        cum_bad = 0.0
        cum_good = 0.0
        for row in resultLocal:
            label_all = float(row['label_all'])
            label_bad = float(row['label_bad'])
            count += label_all
            cum_bad += label_bad
            cum_good = count - cum_bad
            cum_bad_arr.append(cum_bad)
            cum_good_arr.append(cum_good)

        ks = 0.0
        result.append("seq\t评分区间\t订单数\t逾期数\t正常用户数\t百分比\t逾期率\t累计坏账户占比\t累计好账户占比\tKS统计量")
        seq = 0
        length = len(cum_good_arr)
        for row in resultLocal:
            seq += 1
            label_all = float(row['label_all'])
            label_bad = float(row['label_bad'])
            label_good = label_all - label_bad
            min_prob = str(row['min']) if row['min'] is not None else "null"
            max_prob = str(row['max']) if row['max'] is not None else "null"
            range = "[{0}, {1}]".format(min_prob, max_prob)
            pct = round(1000.0 * label_all / count) / 10.0
            overdue = round(1000.0 * label_bad / label_all) / 10.0
            cum_bad_rate = round(1000.0 * cum_bad_arr[seq - 1] / cum_bad_arr[length - 1]) / 10.0
            cum_good_rate = round(1000.0 * cum_good_arr[seq - 1] / cum_good_arr[length - 1]) / 10.0
            gap = round(100.0 * abs(cum_good_rate - cum_bad_rate)) / 100.0
            if ks < gap:
                ks = gap
            result_cur = "%s\t%s\t%s\t%s\t%s\t%s%%\t%s%%\t%s%%\t%s%%\t%s%%"%(str(seq), str(range), str(label_all),str(label_bad),str(label_good),str(pct),str(overdue),str(cum_bad_rate),str(cum_good_rate),str(gap))
            result.append(result_cur)

        return [ks, result]



    def get_src_quantiles(self):
        pass

    def percentile(self):
        pass







