# encoding=utf-8
from sparkxgb import XGBoostEstimator
from evaluation.Evaluator import Evaluator





class XGBoostClassifier():
    def __init__(self, paramMap):
        self.paramMap = paramMap


    def trainAndSave(self):
        pass


    # 加载checkpoint模型继续训练后保存最终模型文件
    def loadModelGoOnTrainAndSave(self):
        pass


    # 保存特征重要性文件到本地
    def saveFeatureImportance(self):
        pass



    # 预测并评估auc
    def predict(self):
        pass



    # 限制树的棵树并预测和评估AUC
    def predictTreeLimit(self):
        pass












