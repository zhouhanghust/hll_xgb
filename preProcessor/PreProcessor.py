# -*- coding: UTF-8 -*-
from params.params import whitelist, cols_non_feature_info
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from hll_xgb.utils.CommonUdf import CommonUdf
from pyspark.ml.feature import VectorAssembler


class PreProcessor():
    nofealist = cols_non_feature_info
    valid_feature = whitelist

    @classmethod
    def transColType(cls, df: 'sparkdf', missing: 'Double')-> 'sparkdf':
        all_df = df
        for colType in all_df.schema:
            if colType.name not in cls.nofealist:
                if isinstance(colType.dataType, StringType):
                    all_df = all_df.withColumn(colType.name, CommonUdf.String2Double(col(colType.name)))
                    print("transform String2Double col"+colType.name+"^______^")

        all_df_fillna = all_df.na.fill(missing) # 可以指定subset进行填充，df.na.fill(missing,subset=['order_id','driver_id'])
        return all_df_fillna


    @classmethod
    def filterDt(cls, df: 'sparkdf', begin: 'string', end: 'string')-> 'sparkdf':
        sel_rows = (col("dt")>=begin) & (col("dt")<=end)
        res = df.filter(sel_rows)
        return res


    @staticmethod
    def selectFeature(df):
        res = df.select(*PreProcessor.valid_feature)
        return res


    @classmethod
    def transVector(cls, df: 'sparkdf', vec_col_name: 'string')-> ('sparkdf', 'array[string]'):
        usefealist = []
        for colType in df.schema:
            if colType.name not in cls.nofealist:
                usefealist.append(colType.name)
        usefealist = [fea for fea in usefealist if fea in cls.valid_feature]
        print("入模特征数："+str(len(usefealist)))

        assembler = VectorAssembler(inputCols=usefealist, outputCol=vec_col_name)
        res = assembler.transform(df)
        outputfeas = cls.nofealist + [vec_col_name]
        res = res.select(*outputfeas)
        return res, usefealist
