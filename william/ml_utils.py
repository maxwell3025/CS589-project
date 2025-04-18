import math
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from collections import Counter


# Load original data

def load_data(path, **kw):
    df = pd.read_csv(path, **kw)

    print(f'\ndf:\n{df}')
    print(f'\ndf.describe():\n{df.describe()}')

    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    return X, y


# Scaler

class MinMaxScaler:
    def __init__(self):
        self.data_min = None
        self.data_max = None
        self.data_range = None

    def fit(self, data_train):
        '''Save training data range info'''
        self.data_min = np.min(data_train, axis=0)
        self.data_max = np.max(data_train, axis=0)
        self.data_range = self.data_max - self.data_min

        # Forcefully set range to 1 if range is zero, to avoid division by zero edge case
        self.data_range[self.data_range == 0] = 1

        # print(f'Data Min: {self.data_min} Max: {self.data_max} Range: {self.data_range}')

    def transform(self, data):
        if self.data_min is None or self.data_max is None:
            raise RuntimeError(
                'MinMaxScaler must be fitted before transforming data!')

        return (data - self.data_min) / self.data_range

    def fit_transform(self, data_train):
        self.fit(data_train)
        return self.transform(data_train)


class CategoryEncoder:
    def __init__(self):
        self.cat_to_idx = None
        self.idx_to_cat = None

    def fit(self, data_train_labels):
        cats = sorted(list(set(data_train_labels)))

        self.cat_to_idx = {v: k for k, v in enumerate(cats)}
        self.idx_to_cat = {k: v for k, v in enumerate(cats)}

    def encode(self, cat):
        if self.cat_to_idx is None:
            raise RuntimeError(
                'CategoryEncoder must be fitted before encoding data')

        return self.cat_to_idx[cat]

    def decode(self, idx):
        if self.idx_to_cat is None:
            raise RuntimeError(
                'CategoryEncoder must be fitted before decoding data')

        return self.idx_to_cat[idx]


# Train test split tools

def stratified_kfold_indices(y, k, verbose=False):
    '''Generate implementation of stratified k-folds'''
    classes, counts = np.unique(y, return_counts=True)

    folds = [[] for i in range(k)]

    for cls, cnt in zip(classes, counts):
        indices = np.where(y == cls)[0]
        np.random.shuffle(indices)

        indices_split = np.array_split(indices, k)

        for i in range(len(folds)):
            folds[i].extend(indices_split[i])

    for fold in folds:
        np.random.shuffle(fold)

    # print(f"    Folds: {folds}")

    # return folds  # Fold indices

    for i in range(k):
        test = folds[i]
        train_folds = [folds[j] for j in range(k) if j != i]
        train = []
        [train.extend(fold) for fold in train_folds]  # Combine training folds

        if verbose:
            print("     k-Fold")
            print(f"     Training:\n{train}")
            print(f"     Testing:\n{test}")

        yield train, test


# Counting

def most_frequent(arr, random_state=None):
    '''Return most frequent item with tie breaking feature'''

    if len(arr) == 0:
        return None

    """
    d = {}
    for item in arr:
        if item not in d:
            d[item] = 1
        else:
            d[item] += 1
    """

    d = Counter(arr)  # Faster

    nmost = max(d.values())

    ties = [k for k, v in d.items() if v == nmost]

    # Tie breaking
    if random_state is None:
        item = random.choice(ties)   # np.random.choice is much slower
    else:
        item = min(ties)   # Pick 'a'  from tied ['b', 'a', 'c']

    return item


# Performance

def calculate_metrics(y_pred, y_test):
    '''Return accuracy, precision, recall, F1'''

    y_pred = np.array(y_pred)
    y_test = np.array(y_test)

    accuracy = np.mean(y_pred == y_test)

    # Get all unique classes from both y_test and y_pred
    classes = np.union1d(np.unique(y_test), np.unique(y_pred))

    precisions = []
    recalls = []
    f1s = []

    for cls in classes:
        tp = np.sum((y_test == cls) & (y_pred == cls))
        fp = np.sum((y_test != cls) & (y_pred == cls))
        fn = np.sum((y_test == cls) & (y_pred != cls))

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        precisions.append(precision)

        recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        recalls.append(recall)

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        f1s.append(f1)

    # Calculate macro averages
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    return accuracy, macro_precision, macro_recall, macro_f1



# Plot
"""
def plot(df_train, df_test, show=False):
    ks = df_train.index

    # Plot
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)

    # df_train.plot()
    plt.errorbar(ks, df_train['mean'], yerr=df_train['std'],
                 fmt='-o', ecolor='black', capsize=5)
    # plt.plot(ks, df_train['accuracy'])
    plt.title('Training Set')
    plt.xlabel('Decision Tree Max Depth')
    plt.ylabel('Accuracy')
    plt.xticks(ks[::2])
    plt.grid(True)

    plt.subplot(1, 2, 2)

    # df_test.plot()
    plt.errorbar(ks, df_test['mean'], yerr=df_test['std'],
                 fmt='-o', ecolor='black', capsize=5)
    # plt.plot(ks, df_test['accuracy'])
    plt.title('Testing Set')
    plt.xlabel('Decision Tree Max Depth')
    plt.ylabel('Accuracy')
    plt.xticks(ks[::2])
    plt.grid(True)

    plt.tight_layout()

    plt.savefig('dt_accuracy.png')

    if show:
        plt.show()


def plot_hist(acc_train, acc_test, bins=100, show=False):
    acc1 = acc_train.iloc[:, -1]

    # Plot
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)

    plt.hist(acc1, bins=bins, edgecolor='black')
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Training Set')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)    # Align label to right
    plt.grid(True)

    plt.subplot(1, 2, 2)

    acc2 = acc_test.iloc[:, -1]

    plt.hist(acc2, bins=bins)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Testing Set')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90, ha='right')    # Align label to right
    plt.grid(True)

    plt.tight_layout()

    plt.savefig('dt_hist.png')

    if show:
        plt.show()
"""


# Information Gain: Gini, Entropy

def entropy(arr):
    '''Return entropy of array distribution'''
    d = Counter(arr)

    counts = np.array(list(d.values()))
    ps = counts / np.sum(counts)

    en = np.sum([-p * math.log(p) for p in ps]) / math.log(2)

    # print(f'    entropy: {en}')

    return en


def gini(arr):
    '''Return Gini coefficient'''
    d = Counter(arr)

    counts = np.array(list(d.values()))
    ps = counts / len(arr)

    return 1 - np.sum(ps ** 2)
