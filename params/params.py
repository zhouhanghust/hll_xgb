# encoding=utf-8
import configparser
import pandas as pd

config = configparser.ConfigParser()
config.read("./config.ini",'utf-8')

XGB_param = dict(config.items("XGBOOST"))
Train_param = dict(config.items("TRAIN"))
ks_detail_out_path=config['FEATURE_ANA']['ks_detail_out_path']
ks_summary_out_path=config['FEATURE_ANA']['ks_summary_out_path']


whitelist = pd.read_csv(Train_param['whitelist_path'], header=None, names['features'])
whitelist = whitelist['features'].values.tolist()
whitelist = [fea.strip() for fea in whitelist]

cols_non_feature_info = ['order_id', 'ab_order', 'driver_id', 'user_id', 'dt', 'label']


