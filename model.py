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
import pickle as pkl


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

def cut_age(x):
    if x < 18:
        return 0
    elif x < 30:
        return 1
    elif x < 40:
        return 2
    elif x < 50:
        return 3
    elif x < 60:
        return 4
    else:
        return 5
# feature
def generate_feature(df):
    df['用户前五个月平均消费值（元）'] = (df['用户近6个月平均消费值（元）'] * 6 - df['用户账单当月总费用（元）']) / 5
    df['当月消费值较前五个月平均消费值'] = df['用户账单当月总费用（元）'] - df['用户前五个月平均消费值（元）']

    # df.loc[df['用户年龄'] == 0, '用户年龄'] = df['用户年龄'].mode()
    df['用户年龄分段'] = df['用户年龄'].apply(cut_age)

    long_tail = ['当月视频播放类应用使用次数', '当月金融理财类应用使用总次数','当月网购类应用使用次数','用户当月账户余额（元）']
    for feature in long_tail:
        threshold = int(df[feature].std()*3 + df[feature].mean())
        df[feature] = df[feature].apply(lambda x: x if x < threshold else threshold)

    app_col = ['当月视频播放类应用使用次数', '当月金融理财类应用使用总次数', '当月网购类应用使用次数']
    df['当月网购类应用使用次数' + '百分比'] = (df['当月网购类应用使用次数']) / (df[app_col].sum(axis=1) + 1e-8)
    # df['用户最近一次缴费距今时长（月）'] = df['缴费用户最近一次缴费金额（元）'].apply(lambda x: 1 if x>0 else 0)

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

    drop_columns = ['用户编码','是否大学生客户','用户实名制是否通过核实',
                '当月是否逛过福州仓山万达', '当月是否到过福州山姆会员店','用户最近一次缴费距今时长（月）']
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

    param1 = {'num_leaves': 40,
             'objective': 'regression_l2',
             'max_depth': 6,
             'learning_rate': 0.005,
             "boosting": "gbdt",
             "feature_fraction": 0.5,
             "bagging_freq": 1,
             "bagging_fraction": 0.5,
             "metric": 'mae',
             "lambda_l1": 0.076,
             "lambda_l2": 0.18,
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
             "lambda_l1": 0.05,
             "lambda_l2": 0.04,
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
             "lambda_l1": 0.145,
             "lambda_l2": 0.042,
             "verbosity": -1,
             'min_data_in_leaf':21}
    param4 = {'num_leaves': 40,
             'objective': 'regression_l1',
             'max_depth': 6,
             'learning_rate': 0.005,
             "boosting": "gbdt",
             "feature_fraction": 0.5,
             "bagging_freq": 1,
             "bagging_fraction": 0.5,
             "metric": 'mae',
             "lambda_l1": 0.1,
             "lambda_l2": 0.095,
             "verbosity": -1}

    param = [param1, param2, param3, param4]
    seed = [2018, 2019, 2018, 2019]
    oof = []
    results = []
    for _param, _seed in zip(param, seed):
        print('*'*50)
        print("seed: {}, type: {}".format(_seed, _param['objective']))
        _oof, _results = _model_main(_param, seed=_seed)
        oof.append(_oof)
        results.append(_results)
    valid = oof[0]*0.251 + oof[1]*0.25 + oof[0]*0.25 + oof[1]*0.25
    print("score :{}, spend:{}".format(score(valid, y_train), time.time() - start))
    final_result = results[0]*0.251 + results[1]*0.25 + oof[0]*0.25 + oof[1]*0.25
    submit(model_name='model_final', predictions=final_result)


def model_fine_tune():
    X_train, y_train, X_test = processing()

    def cache_save():
        with open('output/cache/params.pkl', 'wb') as f:
            pkl.dump(best_params, f)
        with open('output/cache/valid.pkl', 'wb') as f:
            pkl.dump(valid, f)
        with open('output/cache/res.pkl', 'wb') as f:
            pkl.dump(res, f)
    param1 = {'num_leaves': 35,
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
    param2 = {'num_leaves': 40,
              'objective': 'regression_l1',
              'max_depth': 6,
              'learning_rate': 0.005,
              "boosting": "gbdt",
              "feature_fraction": 0.5,
              "bagging_freq": 1,
              "bagging_fraction": 0.5,
              "metric": 'mae',
              "lambda_l1": 0.05,
              "lambda_l2": 0.04,
              "verbosity": -1}

    valid = []
    res = []
    best_params = {}
    for seed in [2018, 2019]:
        print('fire!!!!!')
        print('*' * 50)
        start = time.time()
        clf = my_lgb(folds=5, seed=seed)
        _best_params = clf.optimize_lgb(X_train, y_train, X_test, param1)
        clf.inference_folds(X_train, y_train, X_test, _best_params)

        best_params['{}-mse'.format(seed)] = _best_params
        valid.append(clf.oof)
        res.append(clf.results)
        cache_save()
        del clf
        print('seed:{},training spend {}s'.format(seed, time.time() - start))
        print('*' * 25)
        start = time.time()
        clf = my_lgb(folds=5, seed=seed)
        _best_params = clf.optimize_lgb(X_train, y_train, X_test, param2)
        clf.inference_folds(X_train, y_train, X_test, _best_params)

        best_params['{}-mse'.format(seed)] = _best_params
        valid.append(clf.oof)
        res.append(clf.results)
        cache_save()
        del clf
        print('seed:{},training spend {}s'.format(seed, time.time() - start))
        print('*' * 50)

def merge():
    pass


if __name__=='__main__':
    # model_fine_tune()
    model_final()