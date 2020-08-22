# encoding=utf-8
from sparkxgb import XGBoostEstimator, XGBoostClassificationModel
from evaluation.Evaluator import Evaluator
from preProcessor.PreProcessor import PreProcessor
from pyspark.sql.functions import monotonically_increasing_id



class XGBoostClassifier():
    def __init__(self, paramMap):
        self.paramMap = paramMap

    # 只训练，不保存模型
    def train(self, spark, df, hdfs_model_path):
        print("**************开始训练***************")
        xgb = XGBoostEstimator(**self.paramMap)
        xgbModel = xgb.fit(df)
        print("**************训练完成***************")
        return xgbModel


    def trainAndSave(self, spark, df, hdfs_model_path):
        print("**************开始训练***************")
        print(self.paramMap)
        xgb = XGBoostEstimator(**self.paramMap)
        xgbModel = xgb.fit(df)
        xgbModel.write().overwrite().save(hdfs_model_path)
        print("hdfs模型保存在: ",hdfs_model_path)
        print("**************训练完成***************")
        return xgbModel

    # 保存特征重要性文件到本地
    def saveFeatureImportance(self, traincols, fscore, resPath,train_auc, test_auc, train_ks, test_ks):
        with open(resPath,"w") as f:
            f.write('train AUC: {0}\n'.format(train_auc))
            f.write('test_AUC: {0}\n'.format(test_auc))
            f.write('train ks: {0}\n'.format(train_ks))
            f.write('test_ks: {0}\n'.format(test_ks))
            f.write('\t'.join(['feature_name','feature_score','\n']))
            fscore = sorted(fscore,key=lambda x: x[1],reverse=False)
            for fea_name, fea_rank in fscore:
                f.write('\t'.join([str(fea_name),str(fea_rank),'\n']))

            print("特征重要性文件保存在：",resPath)




    # 预测并评估auc
    def predict(self,spark,tmp,xgb):
        data = PreProcessor.transVector(tmp, 'features')
        predictions = xgb.predict(data,-999).map(lambda row: (row['predictions'][1],row['label']))
        predictions = predictions.toDF("score","label")
        right = predictions.withColumn("idx", monotonically_increasing_id())
        left = tmp.select(['name','idcard','phone']).withColumn("idx",monotonically_increasing_id())
        res_df = left.join(right,['idx'],'inner').drop('idx')
        evaluator_handle = Evaluator(spark)
        auc = evaluator_handle.evaluateAuc(res_df)
        print("AUC: ", auc)
        return res_df, auc




    # 限制树的棵树并预测和评估AUC，若limit=-1表示使用全部棵树
    def predictTreeLimit(self, spark, tmp, xgbModel, treelimit=-1):
        data = PreProcessor.transVector(tmp, 'features')
        predictions = xgbModel.predictTreeLimit(data, -999, treelimit).map(lambda row: (row['predictions'][1],row['label']))
        predictions = predictions.toDF("score", "label")
        right = predictions.withColumn("idx", monotonically_increasing_id())
        left = tmp.select(['name','idcard','phone']).withColumn("idx",monotonically_increasing_id())
        res_df = left.join(right,['idx'],'inner').drop('idx')
        evaluator_handle = Evaluator(spark)
        auc = evaluator_handle.evaluateAuc(res_df)
        print("AUC: ", auc)
        return res_df, auc











