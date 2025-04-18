import sys
import logging
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import Counter
import math

from ml_utils import *

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def Incrementer():
    count = -1

    def _reset():
        nonlocal count
        count = -1

    def _inc():
        nonlocal count
        count += 1
        return count

    return _reset, _inc


resetID, getID = Incrementer()


class DTNode:
    def __init__(self, split_feature=None, split_value=None, label=None, children={}):
        self.split_feature = split_feature
        self.split_value = split_value
        self.label = label     # Classification
        self.children = children
        self.id = getID()

        # print(f'    Create node: {self}')

    def __repr__(self):
        return f'<DTNode: {self.id} label: {self.label} split: {self.split_feature} {self.split_value} children: {self.children}>'


def print_tree(node, indent=0, prefix=""):
    # Leaf node case
    if node.label is not None:
        print(" " * indent + prefix + f"Leaf: {node.label}")
        return

    # Decision node case
    print(" " * indent + prefix + f"Node [ID:{node.id}]")
    print(" " * indent + f"├── Split Feature: {node.split_feature}")

    # Print children with appropriate connectors
    children = list(node.children.items())
    for i, (value, child) in enumerate(children):
        if i == len(children) - 1:
            connector = "└── "
            next_indent = "    "
        else:
            connector = "├── "
            next_indent = "│   "

        print(" " * indent + f"{connector}Branch: {value}")
        print_tree(child, indent + 4, next_indent)


class DecisionTree:
    def __init__(self, min_depth=2, max_depth=100, min_samples=2, min_info_gain=0.001, random_state=None, isBST=False, use_gini=False, early_stop_threshold=1.00):
        self.root = None
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_info_gain = min_info_gain
        self.random_state = random_state
        self.isBST = isBST  # Binary or n-ary tree
        self.use_gini = use_gini
        self.early_stop_threshold = None if early_stop_threshold == 1.00 else early_stop_threshold

    def fit(self, X_train, y_train, skip_normalization=True):
        self.root = self._build_tree(X_train, y_train, depth=0)

        # print(f'\n\nDecision Tree: \n{self.root}\n\n')
        # print_tree(self.root)

    def _build_tree(self, X, y, depth):
        '''Build decision tree by recursion'''

        nlabels = len(np.unique(y))

        # Recursion termination conditions
        stop_criteria = nlabels <= 1
        extra_stop_criteria = depth >= self.max_depth or len(
            y) < self.min_samples

        if self.early_stop_threshold:
            items, counts = np.unique(y, return_counts=True)
            percent = counts[0] / len(y)  # Percent of most common element
            early_stop = percent > self.early_stop_threshold  # e.g., 0.85

            extra_stop_criteria = extra_stop_criteria or early_stop

        # best_feature, cutoff, ig = self._best_split(X, y)
        split_feature, split_value, ig = self._best_split(X, y, self.min_info_gain)

        if stop_criteria or extra_stop_criteria or (split_feature is None and split_value is None):
            label = most_frequent(y, random_state=self.random_state)
            node = DTNode(label=label)

            # print(f'STOP: {node} len(y): {len(y)}')
            return node

        # split_feature = best_feature
        # split_value = cutoff

        uniques = np.unique(X[split_feature])

        if self.isBST is False:     # n-ary tree
            children = {}

            if split_feature is None:
                label = most_frequent(y, random_state=self.random_state)
                return DTNode(label=label)

            for edge in uniques:
                # print(f'      edge: {edge}')
                mask = X[split_feature] == edge

                Xe, ye = X[mask], y[mask]
                child = self._build_tree(Xe, ye, depth+1)
                children.update({edge: child})
                # print(f'      child:\n{child}')

        else:   # Binary tree
            children = {}

            if split_value is None:
                label = most_frequent(y, random_state=self.random_state)
                return DTNode(label=label)

            mask = X[split_feature] <= split_value

            X1, y1 = X[mask], y[mask]
            left = self._build_tree(X1, y1, depth+1)
            children.update({True: left})

            X2, y2 = X[~mask], y[~mask]
            right = self._build_tree(X2, y2, depth+1)
            children.update({False: right})

        return DTNode(split_feature=split_feature, split_value=split_value, children=children)

    def _best_split(self, X, y, min_info_gain=1e-4):

        if not self.use_gini:
            information = entropy
        else:
            information = gini

        original_i = information(y)

        # print(f'  Original info: {original_i}')

        bf, cutoff, ig = None, None, -np.inf

        if self.isBST is False:  # n-ary tree

            for f in X.columns:
                uniques = np.unique(X[f])

                sum_p_i = 0
                for u in uniques:
                    select = X[f] == u
                    if not select.any():
                        continue

                    i = information(y[select])
                    p = len(y[select]) / len(y)
                    sum_p_i += p * i

                gain = original_i - sum_p_i

                # print(f'  {f}: Avg info: {avg}')

                if gain > ig:
                    ig, bf = gain, f

        else:    # Binary tree
            for f in X.columns:
                uniques = np.unique(X[f])
                values = np.sort(uniques)
                midpoints = (values[:-1] + values[1:]) / 2

                for u in midpoints:
                    sum_p_i = 0

                    select = X[f] <= u
                    if not select.any() or select.all():
                        continue

                    # Left
                    i = information(y[select])
                    p = len(y[select]) / len(y)

                    sum_p_i += p * i

                    # Right
                    i = information(y[~select])
                    p = len(y[~select]) / len(y)

                    sum_p_i += p * i

                    gain = original_i - sum_p_i

                    # print(f'  {f}: Avg info: {avg}')

                    if gain > ig:
                        ig, bf, cutoff = gain, f, u

        # print(f'\n    best_split: {bf}, cutoff: {cutoff} Info Gain: {ig}')

        if ig < min_info_gain:
            return None, None, 0
        else:
            return bf, cutoff, ig

        # return bf, cutoff, ig

    def _predict(self, Xi, isTraining=False):
        """
        # Comment out for performance after debugging
        if self.X_train is None or self.y_train is None:
            raise RuntimeError('DecisionTree must be fitted before predictions!')
        """

        # Traverse down the decision tree to see where it lands to its closest sub groups in training dataset, then classify it by majority

        node = self.root

        while node.children:   # Non-leaf node

            if not node.children:  # Leaf node
                return node.label

            value = Xi.get(node.split_feature, None)
            if value is None:
                return node.label

            if self.isBST is False:    # n-ary tree
                edge = value
            else:   # Binary tree
                edge = value <= node.split_value

            next_node = node.children.get(edge, None)
            if next_node is None:
                return node.label

            node = next_node
            # print(f'    ...searching {node}')

        label = node.label
        # print(f'\n\n Stop at node: {node} \nXi: <{Xi}> == Class: {cat}')

        return label

    def predict(self, X, isTraining=False):
        preds = np.array(
            [self._predict(X.iloc[i, :], isTraining=isTraining) for i in range(len(X))])

        # print(f'preds: {preds} length: {len(preds)}')

        return preds

    def accuracy(self, y_actual, y_preds):
        '''Return prediction accuracy'''

        accuracy = np.mean(y_preds == y_actual)

        print(f'accuracy: {accuracy}')
        return accuracy


def run_dt(csv_file_path, n=20, min_depth=2, max_depth=8, skip_normalization=False, isBST=False, use_gini=False, early_stop_threshold=1.00, random_state=None, test_size=0.2):

    # Manage random seeds
    seed_seq = np.random.SeedSequence(random_state)
    split_seed, tree_seed = [c.generate_state(
        n_words=1)[0] for c in seed_seq.spawn(2)]

    X, y = load_data(csv_file_path)

    df_acc_train = pd.DataFrame()
    df_acc_test = pd.DataFrame()

    for depth in range(min_depth, max_depth+1):

        dt = DecisionTree(min_depth=min_depth, max_depth=depth, random_state=tree_seed,
                          isBST=isBST, use_gini=use_gini, early_stop_threshold=early_stop_threshold)

        acc_train = []
        acc_test = []

        for i in range(n):

            resetID()

            dt.root = None

            print(f'\nDT Max Depth: {depth} Round {i}:\n')

            X, y = shuffle(X, y, random_state=random_state)

            # Randomly partition into training set and testing set
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state)

            # Shuffle, randomly partition into train/test, and scale data
            dt.fit(X, y, skip_normalization=skip_normalization)

            # Predictions
            y_train_preds = dt.predict(X_train, isTraining=True)
            acc_train.append(dt.accuracy(y_train, y_train_preds))

            y_test_preds = dt.predict(X_test)
            acc_test.append(dt.accuracy(y_test, y_test_preds))

        df_acc_train[depth] = acc_train
        df_acc_test[depth] = acc_test

    print(f'df_acc_train: \n{df_acc_train}')
    print(f'df_acc_test: \n{df_acc_test}')

    df_train_stats = pd.DataFrame()
    df_train_stats['mean'] = df_acc_train.mean()
    df_train_stats['std'] = df_acc_train.std()

    df_test_stats = pd.DataFrame()
    df_test_stats['mean'] = df_acc_test.mean()
    df_test_stats['std'] = df_acc_test.std()

    print(f'df_train_stats: \n{df_train_stats}')
    print(f'df_test_stats: \n{df_test_stats}')

    # accuracy.append((df_acc_train, df_acc_test))

    # df_train_stats, df_test_stats = accuracy_stats(accuracy)

    plot_hist(df_acc_train, df_acc_test)
    plot(df_train_stats, df_test_stats)

    return dt, df_train_stats, df_test_stats


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='car.csv', help='csv file path, like car.csv')
    parser.add_argument('-n', '--rounds', type=int,
                        default=20, help='number of runs, default 20')
    parser.add_argument('-dmin', '--min_depth', type=int,
                        default=2, help='Min depth of tree, default 2')
    parser.add_argument('-dmax', '--max_depth', type=int,
                        default=8, help='Max depth of tree, default 8')
    parser.add_argument('-igmin', '--min_info_gain', type=float,
                        default=0.001, help='Minimum information gain, default 0.001')
    parser.add_argument('-skip_norm', '--skip_normalization',
                        type=lambda s: s.lower() in ['true', 'y', 'yes', '1'], default=True, help='skip simulation, default True')
    parser.add_argument('-bst', '--isBST', type=lambda s: s.lower()
                        in ['true', 'y', 'yes', '1'], default=False, help='Use numerical splitting, default False')
    parser.add_argument('-gini', '--use_gini', type=lambda s: s.lower()
                        in ['true', 'y', 'yes', '1'], default=False, help='Use Gini coefficient, default False')
    parser.add_argument(
        '-early_stop', '--early_stop_threshold', type=float, default=1.00, help='Early majority stop threshold like 0.85, default 1.00')

    parser.add_argument('-random_state', '--random_state', type=lambda s: int(s) if s.lstrip('-').strip(
    ).isdigit() else None, default=None, help='random seed like 42, -5, default None')

    args, _ = parser.parse_known_args()

    # print(f'args: {args}')

    return args


def main(args):
    run_dt(csv_file_path=args.path, n=args.rounds, min_depth=args.min_depth, max_depth=args.max_depth,
           skip_normalization=args.skip_normalization, isBST=args.isBST, use_gini=args.use_gini, early_stop_threshold=args.early_stop_threshold, random_state=args.random_state)


if __name__ == '__main__':

    args = get_args()
    main(args)
