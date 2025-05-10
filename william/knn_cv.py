import os
import sys
import math
import numpy as np
import pandas as pd
import argparse
import logging
from collections import deque
import subprocess

from sklearn.model_selection import train_test_split

from ml_utils import MinMaxScaler, stratified_kfold_indices, calculate_metrics

from knn import KNN
from random_forest import RandomForest
from nn import NN


VERBOSE = False

if os.environ.get('DEBUG_MODE', False):
    VERBOSE = True


def dprint(msg, verbose=VERBOSE):
    '''Simple debug print (logging)'''
    if verbose:
        print(msg)


def cross_validation(args):
    '''Stratified K-fold validation of NN'''

    # args = get_args()
    
    if args.random_state is not None:
        np.random.seed(args.random_state)

    if args.header is None:
        df = pd.read_csv(args.path, header=None)
    else:
        df = pd.read_csv(args.path)

    print(df.describe())

    print(f'\ndf: \n{df}')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    
    # IMPORTANT: [1,0,1]  to [[1], [0], [1], ...]
    # y = np.array(y).reshape(-1, 1)    # auto rows, exactly one column

    train_result = []
    test_result = []
    ensemble = []

    for fold_i, (train_idx, test_idx) in enumerate(stratified_kfold_indices(y, args.num_kfolds)):

        print(f'\n----- Stratified K-Fold {fold_i + 1} -----')

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        if not args.skip_normalization:
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        dprint(f'\nX_train norm: \n{X_train}')
        dprint(f'X_test norm: \n{X_test}')

        k_range = range(args.kmin, args.kmax, 2)

        model = KNN(k_range=k_range, random_state=args.random_state)
       
        metrics = model.fit(X_train, y_train, exclude_self=args.exclude_self)
        train_result.append(metrics)

        y_preds = model.predict(X_test)
        metrics = model.evaluate(y_test, y_preds)
        test_result.append(metrics)

        ensemble.append(model)

    mean_train_result = sum(train_result) / args.num_kfolds
    mean_test_result = sum(test_result) / args.num_kfolds
    
    print(f'Stratified Results:\n')

    print(f'\nMean Train Metrics: \n{mean_train_result}')
    print(f'\nMean Test Metrics: \n{mean_test_result}')
    
    plot(mean_train_result, 'Training Set', args)
    plot(mean_test_result, 'Testing Set', args)


def plot(df, train_test, args):
    name = args.path.replace('.csv', '')
    fig = df.plot(xlabel='k', grid=True, alpha=0.7, title=train_test).get_figure()
    fig.savefig(f'./img/knn_{name}_{train_test.split()[0].lower()}_{args.exclude_self}.png')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='wdbc_wo_header.csv', help='CSV path od data file, default wdbc_wo_header.csv')
    parser.add_argument('-n', '--rounds', type=int, default=20, help='Rounds for KNN run, default 20')
    parser.add_argument('-kmin', '--kmin', type=int, default=1, help='Min K in KNN, default 1')
    parser.add_argument('-kmax', '--kmax', type=int, default=52, help='Max K for KNN, default 52')
    parser.add_argument('-exclude_self', '--exclude_self',
                        type=lambda s: s.lower() in ['true', 'y', 'yes', '1'], default=False, help='Exclude self in traing, defualt False')
    parser.add_argument('-skip_norm', '--skip_normalization',
                        type=lambda s: s.lower() in ['true', 'y', 'yes', '1'], default=False, help='Skip normalization, default False')
    parser.add_argument('-header', '--header',
                        type=lambda s: s.lower() in ['true', 'y', 'yes', '1'], default=None, help='CSV file header or not, default None')
    parser.add_argument('-kfold', '--num_kfolds', type=int,
                        default=5, help='number of folds for stratified K-Fold, default 5')
    parser.add_argument('-random_state', '--random_state', type=lambda s: int(s) if s.lstrip('-').strip().isdigit() else None, default=None, help='random seed like 42, -5, default None')

    # args = parser.parse_args()
    # 'known' update: so it works on Jupyter Lab as well
    args, _ = parser.parse_known_args()

    # print(f'args: {args}')

    return args


if __name__ == "__main__":
    args = get_args()
    cross_validation(args)
    
