# -*- coding: UTF-8 -*-

import numpy as np
import math

NUMERIC_PATTERN = re.compile(r'^-?(0\.?|[1-9]\d*.\d*$|0\.\d*[1-9]\d*$)|^\.\d*[1-9]\d*$|^-?[1-9]\d*$')

class CalcTools():
    NUMERIC_PATTERN = re.compile(r'^-?(0\.?|[1-9]\d*.\d*$|0\.\d*[1-9]\d*$)|^\.\d*[1-9]\d*$|^-?[1-9]\d*$')

    @staticmethod
    def get_cut_pos(cut_num, vec, head_pos, tail_pos):
        mid_pos = (head_pos + tail_pos) / 2
        if vec[mid_pos] == cut_num and (mid_pos == tail_pos or vec[mid_pos + 1] > cut_num):
            return mid_pos
        elif vec[mid_pos] <= cut_num:
            return get_cut_pos(cut_num, vec, mid_pos + 1, tail_pos)
        else:
            return get_cut_pos(cut_num, vec, head_pos, mid_pos - 1)

    @staticmethod
    def cal_ks(x, sample_num, ks_part=10, dec_pos=3):
        """
        ks计算主函数
        :param x: 数据，二维数组
        :param ks_part: ks分箱数量
        :param dec_pos: 小数点后保留位数
        :return:
        """
        feature_name = x[0]  # 特征名
        feature_data = x[1]
        feature_value = []
        feature_label = []
        length_all = 0  # 样本总数
        for key in feature_data:
            feature_value_tmp = feature_data[key]
            feature_value.extend(feature_value_tmp)
            feature_length_tmp = len(feature_value_tmp)
            feature_label += [key] * feature_length_tmp
            length_all += feature_length_tmp

        hit_index = ~np.isnan(feature_value)  # 判空
        if length_all != sample_num:
            print feature_name
            return {}
        feature_value = np.array(feature_value)
        feature_label = np.array(feature_label)
        feature_hit = feature_value[hit_index]  # 特征命中
        label_hit = feature_label[hit_index]  # 特征命中样本label

        data = np.vstack((label_hit, feature_hit)).T
        sort_ind = np.argsort(data[:, 1])
        data = data[sort_ind]  # 预测值升序

        length = len(feature_hit)  # 命中样本总数
        sum_bad = sum(data[:, 0])  # 命中坏样本总数
        sum_good = length - sum_bad  # 命中好样本总数

        miss_ovd_num = sum(feature_label[~hit_index])  # 特征未命中逾期数
        miss_order_num = length_all - length  # 特征未命中样本数

        sum_bad_all = sum_bad + miss_ovd_num  # 坏样本总数
        sum_good_all = length_all - sum_bad_all  # 好样本总数

        # 特征基本信息
        zero_num = len(feature_hit[feature_hit == 0])
        class_num = len(np.unique(feature_hit))

        # 特征未命中处理
        if length == 0:
            dic_ks = {
                'ks_detail': {
                },
                'basic': {
                    'feature_name': feature_name,
                    'ks': str(np.nan),
                    'iv': str(np.nan),
                    'coverage': '0',
                    'zero_ratio': str(np.nan),
                    'class_num': '0',
                    'avg_val': str(np.nan),
                    'min_val': str(np.nan),
                    'seg_25': str(np.nan),
                    'med': str(np.nan),
                    'seg_75': str(np.nan),
                    'max_val': str(np.nan),
                    'hit_overdue_ratio': str(np.nan),
                    'miss_overdue_ratio': str(round(miss_ovd_num / float(miss_order_num), dec_pos))
                }
            }
            return dic_ks

        if length < 4:
            seg_25, med, seg_75 = [np.nan, np.nan, np.nan]
        else:
            feature_sorted = data[:, 1]
            seg_25, med, seg_75 = feature_sorted[[length / 4, length / 2, length * 3 / 4]]  # 分位点

        hit_overdue_ratio = sum(label_hit) / float(length)  # 命中逾期率

        # 未命中逾期率
        if miss_order_num > 0:
            miss_overdue_ratio = miss_ovd_num / float(miss_order_num)
        else:
            miss_overdue_ratio = np.nan

        # 特征取值唯一处理
        if class_num == 1:
            dic_ks = {
                'ks_detail': {
                },
                'basic': {
                    'feature_name': feature_name,
                    'ks': str(np.nan),
                    'iv': str(np.nan),
                    'coverage': str(round(length / float(length_all), dec_pos)),
                    'zero_ratio': str(round(zero_num / float(length))),
                    'class_num': '1',
                    'avg_val': str(np.nan),
                    'min_val': str(np.min(feature_hit)),
                    'seg_25': str(np.nan),
                    'med': str(np.nan),
                    'seg_75': str(np.nan),
                    'max_val': str(np.max(feature_hit)),
                    'hit_overdue_ratio': str(round(hit_overdue_ratio, dec_pos)),
                    'miss_overdue_ratio': str(round(miss_overdue_ratio, dec_pos))
                }
            }
            return dic_ks

        # 分箱
        cut_list = [0]
        order_num = []
        bad_num = []

        cut_pos_last = -1
        for i in np.arange(ks_part):
            if i == ks_part - 1 or data[length * (i + 1) / ks_part - 1, 1] != data[length * (i + 2) / ks_part - 1, 1]:
                cut_list.append(data[length * (i + 1) / ks_part - 1, 1])
                if i != ks_part - 1:
                    cut_pos = get_cut_pos(
                        data[length * (i + 1) / ks_part - 1, 1],
                        data[:, 1],
                        length * (i + 1) / ks_part - 1,
                        length * (i + 2) / ks_part - 2
                    )
                else:
                    cut_pos = length - 1
                order_num.append(cut_pos - cut_pos_last)
                bad_num.append(sum(data[cut_pos_last + 1:cut_pos + 1, 0]))
                cut_pos_last = cut_pos
        order_num = np.array(order_num)

        # ks计算
        bad_num = np.array(bad_num)
        good_num = order_num - bad_num
        overdue_ratio = np.array([x for x in bad_num * 100 / [float(x) for x in order_num]])
        bad_ratio_sum = np.array([sum(bad_num[:i + 1]) * 100 / float(sum_bad) for i in range(len(bad_num))])
        good_ratio_sum = np.array([sum(good_num[:i + 1]) * 100 / float(sum_good) for i in range(len(good_num))])
        ks_list = abs(good_ratio_sum - bad_ratio_sum)
        ks = max(ks_list)

        # iv计算
        bad_num_iv = list(bad_num)
        good_num_iv = list(good_num)
        # 有未命中样本，需加入空值单独分箱数据
        if miss_order_num > 0:
            bad_num_iv.insert(0, miss_ovd_num)
            good_num_iv.insert(0, miss_order_num - miss_ovd_num)

        bad_num_iv = np.array(bad_num_iv)
        good_num_iv = np.array(good_num_iv)
        bad_ratio = bad_num_iv / float(sum_bad_all)
        good_ratio = good_num_iv / float(sum_good_all)
        woe = map(lambda x: 0 if x == 0 else math.log(x), bad_ratio / good_ratio)
        iv_list = (bad_ratio - good_ratio) * woe
        iv_list = list(iv_list)
        iv = sum(iv_list)
        woe = list(woe)

        # 统计分割区间

        try:
            if dec_pos == 0:
                span_list = ['[%d,%d]' % (int(round(min(data[:, 1]), dec_pos)), int(round(cut_list[1], dec_pos)))]
            else:
                span_list = ['[%s,%s]' % (round(min(data[:, 1]), dec_pos), round(cut_list[1], dec_pos))]
            if len(cut_list) > 2:
                for i in range(2, len(cut_list)):
                    if dec_pos == 0:
                        span_list.append(
                            '(%d,%d]' % (int(round(cut_list[i - 1], dec_pos)), int(round(cut_list[i], dec_pos))))
                    else:
                        span_list.append('(%s,%s]' % (round(cut_list[i - 1], dec_pos), round(cut_list[i], dec_pos)))
        except:
            span_list = ['0']

        # 整理输出使用数据
        order_num = list(order_num)
        bad_num = list(bad_num)
        good_num = list(good_num)
        overdue_ratio = list(overdue_ratio)
        bad_ratio_sum = list(bad_ratio_sum)
        good_ratio_sum = list(good_ratio_sum)
        ks_list = list(ks_list)
        # 有未命中样本，需加入空值单独分箱数据
        if miss_order_num > 0:
            span_list.insert(0, 'Nan')
            order_num.insert(0, miss_order_num)
            bad_num.insert(0, miss_ovd_num)
            good_num.insert(0, miss_order_num - miss_ovd_num)
            overdue_ratio_na = 0 if miss_order_num == 0 else miss_ovd_num * 100 / float(miss_order_num)
            overdue_ratio.insert(0, overdue_ratio_na)
            bad_ratio_sum.insert(0, np.nan)
            good_ratio_sum.insert(0, np.nan)
            ks_list.insert(0, np.nan)
        order_ratio_all = [x * 100 / float(length_all) for x in order_num]
        # 生成输出dict
        dic_ks = {
            'ks_detail': {
                'feature_name': feature_name,
                'iv': iv,
                'ks': ks,
                'span_list': span_list,
                'order_num': order_num,
                'bad_num': bad_num,
                'good_num': good_num,
                'order_ratio': order_ratio_all,
                'overdue_ratio': overdue_ratio,
                'bad_ratio': bad_ratio_sum,
                'good_ratio': good_ratio_sum,
                'ks_list': ks_list,
                'woe': woe,
                'iv_list': iv_list
            },
            'basic': {
                'feature_name': feature_name,
                'ks': str(round(ks, dec_pos)),
                'iv': str(round(iv, dec_pos)),
                'coverage': str(round(length / float(length_all), dec_pos)),
                'zero_ratio': str(round(zero_num / float(length), dec_pos)),
                'class_num': str(class_num),
                'avg_val': str(round(np.mean(feature_hit), dec_pos)),
                'min_val': str(np.min(feature_hit)),
                'seg_25': str(seg_25),
                'med': str(med),
                'seg_75': str(seg_75),
                'max_val': str(np.max(feature_hit)),
                'hit_overdue_ratio': str(round(hit_overdue_ratio, dec_pos)),
                'miss_overdue_ratio': str(round(miss_overdue_ratio, dec_pos))
            }
        }

        return dic_ks

    @staticmethod
    def cal_psi(x, sample_num, psi_part=10, dec_pos=3):
        feature_name = x[0]  # 特征名
        feature_data = x[1]
        data1 = feature_data[0] if 0 in feature_data.keys() else []
        data2 = feature_data[1] if 1 in feature_data.keys() else []
        data1_length = len(data1)
        data2_length = len(data2)
        if (data1_length + data2_length != sample_num) or data1_length * data2_length == 0:
            print feature_name
            return {}
        data1_hit_index = ~np.isnan(data1)
        data2_hit_index = ~np.isnan(data2)
        data1 = np.array(data1)
        data1_hit = data1[data1_hit_index]
        data2 = np.array(data2)
        data2_hit = data2[data2_hit_index]
        data1_hit_length = len(data1_hit)
        data2_hit_length = len(data2_hit)
        data1_hit_ratio = data1_hit_length / float(data1_length)
        data2_hit_ratio = data2_hit_length / float(data2_length)
        if data1_hit_length * data2_hit_length == 0:
            return [feature_name, np.nan, data1_hit_ratio, data2_hit_ratio]
        data1_hit.sort()

        cut_list = [data1_hit[0]]
        order_num = []
        cut_pos_last = -1
        for i in np.arange(psi_part):
            if i == psi_part - 1 or data1_hit[data1_hit_length * (i+1) / psi_part - 1] != data1_hit[data1_hit_length * (i+2) / psi_part - 1]:
                cut_list.append(data1_hit[data1_hit_length * (i+1) / psi_part - 1])
                if i != psi_part - 1:
                    cut_pos = get_cut_pos(
                        data1_hit[data1_hit_length * (i+1) / psi_part - 1],
                        data1_hit,
                        data1_hit_length * (i + 1) / psi_part - 1,
                        data1_hit_length * (i + 2) / psi_part - 2
                    )
                else:
                    cut_pos = data1_hit_length - 1
                order_num.append(cut_pos - cut_pos_last)
                cut_pos_last = cut_pos
        order_num = np.array(order_num)
        order_ratio_1 = order_num / float(data1_hit_length)

        order_num = []
        for i in range(len(cut_list)):
            if i == 0:
                continue
            elif i == 1:
                order_num.append(len(data2_hit[(data2_hit <= cut_list[i])]))
            elif i == len(cut_list) - 1:
                order_num.append(len(data2_hit[(data2_hit > cut_list[i-1])]))
            else:
                order_num.append(len(data2_hit[(data2_hit > cut_list[i - 1]) & (data2_hit <= cut_list[i])]))
        order_num = np.array(order_num)
        order_ratio_2 = order_num / float(data2_hit_length)

        psi = sum([(order_ratio_1[i] - order_ratio_2[i]) * math.log((order_ratio_1[i] / float(order_ratio_2[i])), math.e) if order_ratio_2[i]*order_ratio_1[i]!=0 else np.inf for i in range(len(order_ratio_1))])
        psi = str(round(psi, dec_pos))
        return [feature_name, psi, str(round(data1_hit_ratio, dec_pos)), str(round(data2_hit_ratio, dec_pos))]
