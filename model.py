#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/8 11:40
# @Author  : chuyu zhang
# @File    : model.py
# @Software: PyCharm
import warnings
import time
from model_zoo import my_lgb
import pandas as pd
import numpy as np


def score(pre, truth):
    return 1 / (MAE(pre, truth) + 1)


def MAE(pre, truth):
    return abs((np.rint(pre) - truth)).mean()


def submit(model_name='default', predictions=None):
    sub_df = pd.read_csv('dataset/submit_example.csv')
    sub_df[' score'] = np.rint(predictions).astype(int)
    sub_df.to_csv("output/{}.csv".format(model_name), index=False)

# useless
def percent(df, feature):
    for f in feature:
        df[f] = np.rint(df[f])
        new_feature = f + '_count'
        tmp = df.groupby(f).size().reset_index().rename(columns={0: new_feature})
        df = df.merge(tmp, 'left', on=f)
    return df

# feature
def generate_feature(df):
    df['用户前五个月平均消费值（元）'] = (df['用户近6个月平均消费值（元）'] * 6 - df['用户账单当月总费用（元）']) / 5
    df['当月消费值较前五个月平均消费值'] = df['用户账单当月总费用（元）'] - df['用户前五个月平均消费值（元）']
    app_col = ['当月视频播放类应用使用次数', '当月金融理财类应用使用总次数', '当月网购类应用使用次数']
    df['是否使用网购类应用'] = np.where(df['当月网购类应用使用次数'] > 0, 1, 0)
    df['当月网购类应用使用次数' + '百分比'] = (df['当月网购类应用使用次数']) / (df[app_col].sum(axis=1) + 1e-8)
    df.loc[df['用户年龄'] == 0, '用户年龄'] = df['用户年龄'].mode()
    df['当月视频播放类应用使用次数'] = np.where(df['当月视频播放类应用使用次数'] > 30000, 30000, df['当月视频播放类应用使用次数'])
    df['当月网购类应用使用次数'] = np.where(df['当月网购类应用使用次数'] > 10000, 10000, df['当月网购类应用使用次数'])
    df['当月金融理财类应用使用总次数'] = np.where(df['当月金融理财类应用使用总次数'] > 10000, 10000, df['当月金融理财类应用使用总次数'])
    df['用户当月账户余额（元）'] = np.where(df['用户当月账户余额（元）'] > 2000, 2000, df['用户当月账户余额（元）'])

    return df


def _reduce_mem_usage_(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


def processing():
    train_df = pd.read_csv('dataset/train_dataset.csv')
    test_df = pd.read_csv('dataset/test_dataset.csv')
    train_df = _reduce_mem_usage_(train_df)
    test_df = _reduce_mem_usage_(test_df)
    target = train_df['信用分']
    data = pd.concat([train_df.drop(columns=['信用分']), test_df], axis=0, ignore_index=True)
    data = generate_feature(data)
    train = data.loc[:49999, :]
    test = data.loc[50000:, :]

    drop_columns = ['用户编码']
    X_train = train.drop(columns=drop_columns).values
    y_train = target.values
    X_test = test.drop(columns=drop_columns).values

    return X_train,y_train,X_test


def model_final():
    start = time.time()
    X_train, y_train, X_test = processing()
    def _model_main(param, seed):
        clf = my_lgb(folds=5, seed=seed)
        clf.inference_folds(X_train, y_train, X_test, param)
        return clf.oof, clf.results
    """
    param1 = {'num_leaves': 40,
             'objective':'regression_l2',
             'max_depth': 6,
             'learning_rate': 0.005,
             "boosting": "gbdt",
             "feature_fraction": 0.5,
             "bagging_freq": 1,
             "bagging_fraction": 0.5,
             "metric": 'mae',
             "lambda_l1": 5,
             "lambda_l2": 0,
             "verbosity": -1}
    param2 = {'num_leaves': 35,
             'objective': 'regression_l2',
             'max_depth': 6,
             'learning_rate': 0.005,
             "boosting": "gbdt",
             "feature_fraction": 0.5,
             "bagging_freq": 1,
             "bagging_fraction": 0.5,
             "metric": 'mae',
             "lambda_l1": 10,
             "lambda_l2": 1,
             "verbosity": -1}
    param3 = {'num_leaves': 35,
             'objective': 'regression_l1',
             'max_depth': 6,
             'learning_rate': 0.005,
             "boosting": "gbdt",
             "feature_fraction": 0.5,
             "bagging_freq": 1,
             "bagging_fraction": 0.5,
             "metric": 'mae',
             "lambda_l1": 5,
             "lambda_l2": 1,
             "verbosity": -1}
    param4 = {'num_leaves': 37,
             'objective': 'regression_l1',
             'max_depth': 6,
             'learning_rate': 0.005,
             "boosting": "gbdt",
             "feature_fraction": 0.5,
             "bagging_freq": 1,
             "bagging_fraction": 0.5,
             "metric": 'mae',
             "lambda_l1": 5,
             "lambda_l2": 0,
             "verbosity": -1}
    """
    param1 = {'num_leaves': 40,
             'objective': 'regression_l2',
             'max_depth': 6,
             'learning_rate': 0.005,
             "boosting": "gbdt",
             "feature_fraction": 0.5,
             "bagging_freq": 1,
             "bagging_fraction": 0.5,
             "metric": 'mae',
             "lambda_l1": 5,
             "lambda_l2": 0,
             "verbosity": -1}
    param2 = {'num_leaves': 40,
             'objective': 'regression_l2',
             'max_depth': 6,
             'learning_rate': 0.005,
             "boosting": "gbdt",
             "feature_fraction": 0.5,
             "bagging_freq": 1,
             "bagging_fraction": 0.5,
             "metric": 'mae',
             "lambda_l1": 5,
             "lambda_l2": 1,
             "verbosity": -1}
    param3 = {'num_leaves': 40,
             'objective': 'regression_l1',
             'max_depth': 6,
             'learning_rate': 0.005,
             "boosting": "gbdt",
             "feature_fraction": 0.5,
             "bagging_freq": 1,
             "bagging_fraction": 0.5,
             "metric": 'mae',
             "lambda_l1": 5,
             "lambda_l2": 1,
             "verbosity": -1}
    param4 = {'num_leaves': 40,
             'objective': 'regression_l1',
             'max_depth': 6,
             'learning_rate': 0.005,
             "boosting": "gbdt",
             "feature_fraction": 0.5,
             "bagging_freq": 1,
             "bagging_fraction": 0.5,
             "metric": 'mae',
             "lambda_l1": 5,
             "lambda_l2": 0,
             "verbosity": -1}

    param = [param1, param2, param3, param4]
    seed = [2018, 2019, 2018, 2019]
    oof = []
    results = []
    for _param, _seed in zip(param, seed):
        print('*'*50)
        print("seed: {}, type: {}".format(_seed, _param['metric']))
        _oof, _results = _model_main(_param, seed=_seed)
        oof.append(_oof)
        results.append(_results)
    valid = oof[0]*0.25 + oof[1]*0.25 + oof[2]*0.25 + oof[3]*0.25
    print("score :{}, spend:{}".format(score(valid, y_train), time.time() - start))
    final_result = results[0]*0.25 + results[1]*0.25 + results[2]*0.25 + results[3]*0.25
    submit(model_name='my_model', predictions=final_result)


if __name__=='__main__':
    model_final()