# -*- coding: UTF-8 -*-
from hll_xgb.dataAnalyzer.run_ks_psi import Run_ks_psi




class Evaluator():
    def __init__(self, spark: 'SparkSession'):
        self._spark = spark


    def evaluateConfusionMatrix(self, df: 'sparkdf'):
        pass

    def evaluateAuc(self, df: 'sparkdf'):
        pass


    def evaluateKsCustomized(self, predictions: 'sparkdf', prob ="score"):
        ks_info = Run_ks_psi.getKsInfo(predictions, prob, 10, 3)








