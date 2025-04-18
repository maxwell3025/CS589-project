import sys
import logging
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from ml_utils import *


class KNN:
    def __init__(self, k_range=range(1, 52, 2), random_state=None):
        self.original_data = None   # Original data
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.k_range = k_range
        self.random_state = random_state

    def load_data(self, path, **kw):
        df = pd.read_csv(path, **kw)
        self.original_data = df.values

        print(f'{df}')
        print(f'{df.describe()}')

    def fit(self, test_size=0.2, skip_normalization=False):
        '''Shuffle, random partition, and scale data'''

        # Shuffle data
        data = self.original_data
        X, y = data[:, :-1], data[:, -1]   # Last column is class/label
        X, y = my_shuffle(X, y)
        # X, y = shuffle(X, y)

        # Randomly partition into training set and testing set
        X_train, X_test, y_train, y_test = my_train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=None)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=None)

        # Feature scale (AI works on normalized data)
        if skip_normalization:
            print(
                f'\n\n***** WARNING: Feature normalization is skipped! It is generally required! *****\n\n')
            self.X_train = X_train
            self.X_test = X_test
        else:
            scaler = MinMaxScaler()  # My implementation below
            self.X_train = scaler.fit_transform(X_train)
            self.X_test = scaler.transform(X_test)

        # Encode classes/labels into integers
        # Update: This is computationally more efficient, but not strictly needed, since modified accuracy() can compare classes/labels (not just numerical integers).
        encoder = CategoryEncoder()  # My implementation below
        encoder.fit(y_train)
        self.y_train = np.array([encoder.encode(yi) for yi in y_train])
        self.y_test = np.array([encoder.encode(yi) for yi in y_test])

    def _prediction(self, Xi, isTraining=False):
        """
        # Comment out for performance
        if self.X_train is None or self.y_train is None:
            raise RuntimeError('KNN must be fitted before predictions!')
        """

        dist = distance_vec(Xi, self.X_train)

        idx_sorted = np.argsort(dist)
        # dist_sorted = np.sort(dist)
        dist_sorted = dist[idx_sorted]
        y_train_sorted = self.y_train[idx_sorted]

        # print(f'Training: {isTraining} dist_sorted: {dist_sorted[:10]} mean dist: {dist_sorted.mean()}')

        if isTraining:
            # Fixed nonzero dist_sorted[0] problem by refactoring code:
            # Remove all global variables (unintentionally coupled),
            # use class and functions instead.

            # print(f'Xi: {Xi} {Xi in self.X_train} {dist_sorted[0]}')

            # Confidence check
            assert dist_sorted[0] == 0, 'Assert error: expect dist_sorted[0] == 0'
        pred = []
        for k in self.k_range:
            if isTraining:
                # Exclude self vs self (distance = 0)
                cat = most_frequent(
                    y_train_sorted[1: k+1], random_state=self.random_state)
            else:
                cat = most_frequent(
                    y_train_sorted[: k], random_state=self.random_state)

            pred.append(cat)

        return np.array(pred)

    def prediction(self, X, isTraining=False):
        y_preds = np.empty((len(X), len(self.k_range)))

        for i in range(len(X)):
            y_preds[i] = self._prediction(X[i], isTraining=isTraining)

        return y_preds

    def accuracy(self, y_actual, y_preds):
        '''Evaluate kNN accuracy vs k values'''

        """
        # Correct but slower
        p = np.zeros(len(self.k_range))
        for j in range(len(self.k_range)):
            diff = y_preds[:, j] - y_actual
            incorrect_num = np.count_nonzero( diff )
            # print(f'k={ks[j]}: len(diff)={len(diff)} diff = {diff} incorrect: {incorrect_num}')
            p[j] = 1.0 - incorrect_num / len(y_actual)
        """

        # Convert: like  [1, 0, 1, ...] to [[1], [0], [1], ...]
        # y_actual_ = y_actual[:, None]
        y_actual_ = np.array([[x] for x in y_actual])

        accuracy = np.mean(y_preds == y_actual_, axis=0)

        df = pd.DataFrame({'accuracy': accuracy}, index=self.k_range)

        print(f'{df}')

        return df


# Equivalent implementation of sklearn.utils.shuffle
def my_shuffle(X, y):
    '''Randomly shuffle data to remove ordering'''

    idx_shuffled = np.random.permutation(len(y))

    return X[idx_shuffled], y[idx_shuffled]


def my_train_test_split(X, y, test_size=0.2, stratify=None, random_state=None):
    '''Randomly partition data into a training set and a testing set for a given ratio and retaining same classes' distributions in the training and testing sets'''

    # dict: histogram of y unique values and counts
    if stratify is None:
        stratify = y

    np.random.seed(random_state)

    d = {}
    for x in stratify:
        if x not in d:
            d[x] = 1   # First occurrence
        else:
            d[x] += 1

    idx = range(len(stratify))

    # Get indexes for each unique value (class/label) in stratify
    d_stratify = {x: [] for x in d.keys()}
    for i in idx:
        x = stratify[i]
        d_stratify[x].append(i)

    # Pick test data in random, yet keep statistic distribution
    idx_test = []
    for x in d.keys():
        idx_picked = list(np.random.choice(
            d_stratify[x], (int)(test_size * len(d_stratify[x]))))
        idx_test.extend(idx_picked)

    idx_train = [i for i in idx if i not in idx_test]

    idx_combined = idx_test + idx_train
    # Re-order: test part first, then train part
    X1 = X[idx_combined]
    y1 = y[idx_combined]

    X_test = X1[: len(idx_test)]
    y_test = y1[: len(idx_test)]
    X_train = X1[len(idx_test):]
    y_train = y1[len(idx_test):]

    # Shuffle
    X_train, y_train = my_shuffle(X_train, y_train)
    X_test, y_test = my_shuffle(X_test, y_test)

    return X_train, X_test, y_train, y_test


def distance(Xi, Xj):
    '''Euclidean distance between two n-dimensional points'''
    return np.sqrt(np.sum((Xi - Xj) ** 2))


def distance_vec(Xi, X_train_data):
    return np.sqrt(np.sum((X_train_data - Xi) ** 2, axis=1))


def run_knn(csv_file_path, n=20, k_range=range(1, 52, 2), skip_normalization=False):
    knn = KNN(k_range=k_range)
    knn.load_data(csv_file_path, header=None)

    accuracy = []

    for i in range(n):

        print(f'\nRound {i}:\n')

        # Shuffle, random partition, and scale data
        knn.fit(test_size=0.2, skip_normalization=skip_normalization)

        # Predictions

        y_train_preds = knn.prediction(knn.X_train, isTraining=True)
        df_train_accuracy = knn.accuracy(knn.y_train, y_train_preds)

        y_test_preds = knn.prediction(knn.X_test)
        df_test_accuracy = knn.accuracy(knn.y_test, y_test_preds)

        accuracy.append((df_train_accuracy, df_test_accuracy))

    df_train, df_test = accuracy_stats(accuracy)

    plot(df_train, df_test)

    return knn, accuracy, df_train, df_test


def accuracy_stats(accuracy):
    acc_train = pd.concat([accuracy[i][0]
                          for i in range(len(accuracy))], axis=1)
    acc_test = pd.concat([accuracy[i][1]
                         for i in range(len(accuracy))], axis=1)

    acc_train = acc_train.T
    acc_test = acc_test.T

    print(f'{acc_train}')
    print(f'{acc_test}')

    stats_train = pd.DataFrame(index=['mean', 'high', 'low', 'std'])
    stats_test = pd.DataFrame(index=['mean', 'high', 'low', 'std'])

    for col in acc_train.columns:
        stats_train[col] = [np.mean(acc_train[col]), np.max(
            acc_train[col]), np.min(acc_train[col]), acc_train[col].std()]
        stats_test[col] = [np.mean(acc_test[col]), np.max(
            acc_test[col]), np.min(acc_test[col]), acc_test[col].std()]

    print(f'stats_train:\n{stats_train.T}')
    print(f'stats_test:\n{stats_test.T}')

    return stats_train.T, stats_test.T


def plot(df_train, df_test, show=False):
    ks = df_train.index

    # Plot
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)

    # df_train.plot()
    plt.errorbar(ks, df_train['mean'], yerr=df_train['std'],
                 fmt='-o', ecolor='black', capsize=5)
    plt.title('Training Set')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.xticks(ks[::2])
    plt.grid(True)

    plt.subplot(1, 2, 2)

    # df_test.plot()
    plt.errorbar(ks, df_test['mean'], yerr=df_test['std'],
                 fmt='-o', ecolor='black', capsize=5)
    plt.title('Testing Set')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.xticks(ks[::2])
    plt.grid(True)

    plt.tight_layout()

    plt.savefig('knn_accuracy.png')

    if show:
        plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='wdbc_wo_header.csv')
    parser.add_argument('-n', '--rounds', type=int, default=20)
    parser.add_argument('-kmin', '--kmin', type=int, default=1)
    parser.add_argument('-kmax', '--kmax', type=int, default=52)
    parser.add_argument('-skip_norm', '--skip_normalization',
                        type=lambda s: s.lower() in ['true', 'y', 'yes', '1'], default=False)

    # args = parser.parse_args()
    # 'known' update: so it works on Jupyter Lab as well
    args, _ = parser.parse_known_args()

    # print(f'args: {args}')

    return args


def main(args):
    k_range = range(args.kmin, args.kmax, 2)

    run_knn(args.path, n=args.rounds, k_range=k_range,
            skip_normalization=args.skip_normalization)


if __name__ == '__main__':
    args = get_args()
    main(args)
