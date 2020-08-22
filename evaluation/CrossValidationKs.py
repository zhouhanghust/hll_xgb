# encoding=utf-8
from classification.XGBoostClassifier import XGBoostClassifier
from evaluation.Evaluator import Evaluator
from params.params import config
import numpy as np


class CrossValidationKs():
    def __init__(self, spark, train, cv_num):
        self.spark = spark
        self.train = train
        self.cv_num = cv_num

    def crossValidationByXGB(self, paramMap):
        split_num_arr = []
        for i in range(self.cv_num):
            split_num_arr.append(1.0 / self.cv_num)
        split_data_arr = self.train.randomSplit(split_num_arr, seed=666)
        split_data_arr.map(lambda x: x.cache())
        kfold_res = self.KFold(split_data_arr)
        train_ks_list = []
        val_ks_list = []
        train_auc_list = []
        val_auc_list = []

        for i in range(len(kfold_res)):
            train_df, validation_df = kfold_res[i]
            xgb_handle = XGBoostClassifier(config['XGBOOST'])
            xgbModel = xgb_handle.train(train_df)
            train_res, train_auc = xgb_handle.predict(self.spark, train_df, xgbModel)
            validation_res, validation_auc = xgb_handle.predict(self.spark, validation_df, xgbModel)

            evaluator_handler = Evaluator(self.spark)
            train_ks = evaluator_handler.evaluateKsCustomized(train_res, "score")
            val_ks = evaluator_handler.evaluateKsCustomized(validation_res,"score")
            print("fold"+str(i)+" train ks: {0}".format(train_ks))
            print("fold" + str(i) + " val ks: {0}".format(val_ks))
            print("fold" + str(i) + " train auc: {0}".format(train_auc))
            print("fold" + str(i) + " val auc: {0}".format(validation_auc))

            train_ks_list.append(train_ks)
            train_auc_list.append(train_auc)
            val_ks_list.append(val_ks)
            val_auc_list.append(validation_auc)

        print('----------------avg----------------')
        print("train average ks is ", np.mean(train_ks_list))
        print("val average ks is ", np.mean(val_ks_list))
        print("train average auc is ", np.mean(train_auc_list))
        print("val average auc is ", np.mean(val_auc_list))


    def KFold(self, arr_df):
        train_df = self.spark.emptyDataFrame
        validation_df = self.spark.createDataFrame(self.spark.sparkContext.emptyRDD(), arr_df[0].schema)
        kfold_res = []
        for i in range(len(arr_df)):
            validation_df = arr_df[i]
            flag = True
            for j in range(len(arr_df)):
                if flag and j!=i:
                    train_df = arr_df[j]
                elif not flag and j!=i:
                    train_df = train_df.union(arr_df[j])
                if j!=i:
                    flag = True
            kfold_res.append([train_df, validation_df])

        return kfold_res