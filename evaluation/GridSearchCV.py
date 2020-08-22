# encoding=utf-8
from sparkxgb import XGBoostEstimator, XGBoostClassificationModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidation, ParamGridBuilder

class GridSearchCV():

    @staticmethod
    def xgbprocesscv(trainData, paramMap):
        xgbEstimator = XGBoostEstimator(**paramMap)
        paramGrid = ParamGridBuilder()\
            .addGrid(xgbEstimator.maxDepth, [5,6,7])\
            .addGrid(xgbEstimator.eta, [0.2,0.3])\
            .addGrid(xgbEstimator.round, [100])\
            .build()

        evaluator = BinaryClassificationEvaluator()\
            .setLabelCol("label")\
            .setRawPredictionCol("probabilities")\
            .setMetricName("areaUnderROC")

        cv = CrossValidation()\
            .setEstimator(xgbEstimator)\
            .setEvaluator(evaluator)\
            .setEstimatorParamMaps(paramGrid)\
            .setNumFold(5)

        cvmodel = cv.fit(trainData)
        print("============== best model param ==============")
        bestModelPm = cvmodel.bestModel.parent.extractParamMap()
        print(bestModelPm)
        bestModel = cvmodel.bestModel
        return bestModel





