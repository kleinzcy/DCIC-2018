#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/12 14:39
# @Author  : chuyu zhang
# @File    : model_zoo.py
# @Software: PyCharm


import lightgbm as lgb
from sklearn.model_selection import KFold
import xgboost as xgb

from bayes_opt import BayesianOptimization

import numpy as np
import pandas as pd

import time

class my_lgb:
    def __init__(self, folds=5, seed=0):
        self.results = None
        self.feature_importance = []
        self.folds = folds
        self.seed = seed
        self.oof = None

    def inference_folds(self, X_train, y_train, X_test, param, cv=False):
        # 五折交叉验证
        folds = KFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        oof = np.zeros(X_train.shape[0])
        predictions = np.zeros(X_test.shape[0])

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
            print("fold n°{}".format(fold_ + 1))
            trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
            val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

            num_round = 10000
            clf = lgb.train(param,
                            trn_data,
                            num_round,
                            valid_sets=[trn_data, val_data],
                            verbose_eval=200,
                            early_stopping_rounds=100)
            oof[val_idx] = np.rint(clf.predict(X_train[val_idx], num_iteration=clf.best_iteration))

            if not cv:
                predictions += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

            self.feature_importance.append(clf.feature_importance())

        if not cv:
            self.results = np.rint(predictions).astype('int64')

        self.oof = oof

        print("score: {:.8f}, MAE: {}".format(self.score(oof, y_train), self.MAE(oof, y_train)))

        return self.score(self.oof, y_train)

    def score(self, pre, truth):
        return 1 / (self.MAE(pre, truth) + 1)

    @staticmethod
    def MAE(pre, truth):
        return abs((pre.astype(int) - truth)).mean()

    def submit(self, example_dir='dataset/submit_example.csv',output_name = 'default'):
        sub_df = pd.read_csv(example_dir)
        sub_df[' score'] = self.results
        sub_df.to_csv("output/{}.csv".format(output_name), index=False)

    def importance_feature(self, feature_name):
        _importance_feature = np.array(self.feature_importance).mean(axis=0).astype(int)
        return pd.DataFrame({'score':_importance_feature}, index=feature_name).sort_values(by='score', ascending=False)

    def optimize_lgb(self, X_train, y_train, X_test, general_params):
        # 'min_data_in_leaf': (10, 50),
        # 'bagging_fraction': (0.3, 1.0),
        #  'feature_fraction': (0.3, 1.0),
        params ={
            'num_leaves': (30, 40),
            'lambda_l1': (5, 10),
            'lambda_l2': (0, 3)
        }

        def lgb_cv(num_leaves, lambda_l1, lambda_l2):
            param = {
                # general parameters
                'objective': general_params['objective'],
                'boosting': general_params['boosting'],
                'metric': general_params['metric'],
                'bagging_freq': general_params['bagging_freq'],
                'learning_rate': general_params['learning_rate'],
                'verbosity': general_params["verbosity"],
                'max_depth': general_params['max_depth'],
                # tuning parameters
                'num_leaves': int(num_leaves),
                'bagging_fraction': 0.5,
                'feature_fraction': 0.5,
                'lambda_l1': lambda_l1,
                'lambda_l2': lambda_l2
            }

            return self.inference_folds(X_train, y_train, X_test, param)

        lgbBO = BayesianOptimization(lgb_cv, params)

        start_time = time.time()
        lgbBO.maximize(init_points=5, n_iter=10)
        end_time = time.time()
        print("Final result:{}, spend {}s".format(lgbBO.max, start_time-end_time))
        best_params = lgbBO.max['params']
        best_params['objective'] = general_params['objective']
        best_params['boosting'] = general_params['boosting']
        best_params['metric'] = general_params['metric']
        best_params['bagging_freq'] = general_params['bagging_freq']
        best_params['learning_rate'] = general_params['learning_rate']
        best_params['max_depth'] = general_params['max_depth']
        best_params['verbosity'] = general_params['verbosity']
        best_params['feature_fraction'] = 0.5
        best_params['bagging_fraction'] = 0.5
        best_params['num_leaves'] = int(best_params['num_leaves'])
        """
        best_params['lambda_l1'] = int(best_params['lambda_l1'])
        best_params['lambda_l2'] = int(best_params['lambda_l2'])
        """


        return best_params


class my_xgb:
    def __init__(self, folds=5):
        self.results = None
        self.feature_importance = []
        self.folds = folds

    def inference_folds(self, X_train, y_train, X_test, param):
        # 五折交叉验证
        folds = KFold(n_splits=self.folds, shuffle=True, random_state=2019)
        oof = np.zeros(X_train.shape[0])
        predictions = np.zeros(X_test.shape[0])

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
            print("fold n°{}".format(fold_ + 1))
            trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
            val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])

            watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
            clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                            verbose_eval=100, params=param)
            oof[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
            predictions += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits

            self.feature_importance.append(clf.feature_importance())

        self.results = predictions


        print("score: {:.8f}, MAE: {}".format(self.score(oof, y_train), self.MAE(oof, y_train)))

    def score(self, pre, truth):
        return 1 / (self.MAE(pre, truth) + 1)

    def MAE(self, pre, truth):
        return abs((pre.astype(int) - truth)).mean()

    def submit(self, example_dir='dataset/submit_example.csv', output_name = 'default'):
        sub_df = pd.read_csv(example_dir)
        sub_df[' score'] = np.rint(self.results).astype('int64')
        sub_df.to_csv("output/{}.csv".format(output_name), index=False)

    def importance_feature(self, feature_name):
        _importance_feture = np.array(self.feature_importance).mean(axis=0).astype(int)
        return pd.DataFrame({'score':_importance_feture}, index=feature_name).sort_values(by='score', ascending=False)
