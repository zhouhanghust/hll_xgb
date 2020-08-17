# -*- coding: UTF-8 -*-
from hll_xgb.dataAnalyzer.utils.tools_ks_psi import CalcTools
import re
import sys
import os
import pandas as pd
import numpy as np
from hll_xgb.config import cols_non_feature_info, cols_feature_all


class Run_ks_psi():
    invalid_feature = cols_non_feature_info
    valid_feature = cols_feature_all

    @staticmethod
    def getKsInfo(df: 'sparkdf', probName: 'string', ks_part=10, dec_pos=3):
        def _data_to_list(a):
            return {a[1]: [a[0]]}

        def _data_append(a, b):
            if b[1] in a.keys():
                a[b[1]].append(b[0])
            else:
                a[b[1]] = [b[0]]
            return a

        def _data_extend(a, b):
            for key in b.keys():
                if key in a.keys():
                    a[key].extend(b[key])
                else:
                    a[key] = b[key]
            return a

        def _transform_data(row, feature_name):
            out_list = []
            label = int(row['label'])
            for index, value in enumerate(feature_name):
                if value != 'label':
                    feature_tmp = row[index]
                    if CalcTools.NUMERIC_PATTERN.match(str(feature_tmp)):
                        feature = float(feature_tmp)
                        out_list.append((value, [feature, label]))
                    elif feature_tmp in CalcTools.DEFAULT_NAN:
                        feature = np.nan
                        out_list.append((value, [feature, label]))
            return out_list

        feature_name = [name for name in df.columns if name in Run_ks_psi.valid_feature+[probName]]
        drop_feature = [i for i in Run_ks_psi.invalid_feature if i in feature_name]
        if 'label' in drop_feature:
            drop_feature.remove('label')
        df = df.drop(*drop_feature).cache()
        sample_num = df.count()
        feature_name = df.columns
        if 'label' not in feature_name:
            print('No Label')
            sys.exit(1)
        # ks计算
        result = df.rdd.flatMap(
            lambda x: _transform_data(x, feature_name)
        ).combineByKey(_data_to_list, _data_append, _data_extend).map(
            lambda row: CalcTools.cal_ks(row, sample_num, ks_part=ks_part, dec_pos=dec_pos)).collect()

        return result




    @staticmethod
    def getPsi(df, psi_part=10, dec_pos=3):
        def _data_to_list(a):
            return {a[1]: [a[0]]}

        def _data_append(a, b):
            if b[1] in a.keys():
                a[b[1]].append(b[0])
            else:
                a[b[1]] = [b[0]]
            return a

        def _data_extend(a, b):
            for key in b.keys():
                if key in a.keys():
                    a[key].extend(b[key])
                else:
                    a[key] = b[key]
            return a

        def _transform_data(row, feature_name):
            out_list = []
            for index, value in enumerate(feature_name):
                sample_split = int(row['sample_split'])
                if value != 'sample_split':
                    feature_tmp = row[index]
                    if feature_tmp in CalcTools.DEFAULT_NAN:
                        feature = np.nan
                        out_list.append((value, [feature, sample_split]))
                    elif CalcTools.NUMERIC_PATTERN.match(str(feature_tmp)):
                        feature = float(feature_tmp)
                        out_list.append((value, [feature, sample_split]))
            # todo 增加保存到本地的功能
            return out_list

        feature_name = [name for name in df.columns if name in Run_ks_psi.valid_feature]
        drop_feature = [i for i in Run_ks_psi.invalid_feature if i in feature_name]
        df = df.drop(*drop_feature).cache()
        sample_num = df.count()
        feature_name = df.columns
        result = df.rdd.flatMap(
            lambda x: _transform_data(x, feature_name)
        ).combineByKey(_data_to_list, _data_append, _data_extend).map(
            lambda row: CalcTools.cal_psi(row, sample_num, psi_part=psi_part, dec_pos=dec_pos)).collect()
        psi_list = []
        for i in result:
            if i:
                psi_list.append(i)
        psi_df = pd.DataFrame(psi_list, columns=['feature_name', 'psi', 'sample1_hit_ratio', 'sample2_hit_ratio'])

        # todo 增加保存到本地的功能
        return psi_df



