# -*- coding: UTF-8 -*-
from hll_xgb.dataAnalyzer.utils.tools_ks_psi import CalcTools
import re
import sys
import os
import pandas as pd
import numpy as np
from params.params import cols_non_feature_info, whitelist


class Run_ks_psi():
    invalid_feature = cols_non_feature_info
    valid_feature = whitelist

    @staticmethod
    def getKsInfo(df: 'sparkdf', probName: 'string', ks_part=10, dec_pos=3, calc_specific=[], ks_detail_out_path='', ks_summary_out_path=''):
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

        if calc_specific:
            feature_name = calc_specific + [probName]
        else:
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

        basic_list = []

        with open(ks_detail_out_path, 'w') as f_ks:
            for fea_item in result:
                if fea_item:
                    basic_list.append(fea_item['basic'])
                    ks_info = fea_item['ks_detail']
                    if ks_info:
                        fea = ks_info['feature_name']
                        f_ks.write(fea + '\nks: %.2f%%\niv: %.4f\t\t\t\t\t\t\t\t\t\t\t\t\n' % (
                            ks_info['ks'], ks_info['iv']))
                        f_ks.write('\t'.join(
                            ['seq',
                             '评分区间',
                             '订单数',
                             '逾期数',
                             '正常用户数',
                             '百分比(%)',
                             '逾期率(%)',
                             '累计坏账户占比(%)',
                             '累计好账户占比(%)',
                             'KS统计量(%)',
                             'WOE',
                             'IV统计量']) + '\n')
                        for i in range(len(ks_info['ks_list'])):
                            f_ks.write('%d\t%s\t%d\t%d\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f\n' % (
                                i + 1, ks_info['span_list'][i], ks_info['order_num'][i], ks_info['bad_num'][i],
                                ks_info['good_num'][i],
                                ks_info['order_ratio'][i], ks_info['overdue_ratio'][i], ks_info['bad_ratio'][i],
                                ks_info['good_ratio'][i], ks_info['ks_list'][i], ks_info['woe'][i],
                                ks_info['iv_list'][i]))

        basic_df = pd.DataFrame(basic_list)
        with open(ks_summary_out_path, 'wb') as f:
            basic_df.loc[:, [
                                'feature_name', 'ks', 'iv', 'coverage', 'zero_ratio', 'class_num',
                                'avg_val', 'min_val', 'seg_25', 'med', 'seg_75', 'max_val',
                                'hit_overdue_ratio', 'miss_overdue_ratio'
                            ]].to_csv(f, sep='\t', index=False)


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

        return psi_df



