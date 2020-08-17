# -*- coding: UTF-8 -*-
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType, IntegerType



class UdfTool():

    @staticmethod
    def String2Double():
        def s2d(data: 'string'):
            if data is None or data == '':
                return -999.0
            elif data.lower() == 'nan':
                return -999.0
            else:
                return float(data)
        return udf(s2d, DoubleType())


    @staticmethod
    def String2Int():
        def s2i(data: 'string'):
            if data is None or data == '':
                return 0
            elif data.lower() == 'nan':
                return 0
            else:
                return int(data)

        return udf(s2i, IntegerType())



class CommonUdf():
    String2Double = UdfTool.String2Double()
    String2Int = UdfTool.String2Int()
