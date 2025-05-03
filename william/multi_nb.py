import sys
import logging
import random
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from ml_utils import preprocess_text, load_training_set, load_test_set


class MultiNB:
    def __init__(self, alpha=0, use_log_pr=True):
        self.vocab = set()
        self.alpha = alpha
        self.use_log_pr = use_log_pr
        self.count_class = defaultdict(int)      # e.g., {'positive': 42, 'negative': 35}
        self.count_class_word = defaultdict(lambda: defaultdict(int))  # e.g., {'positive': {'like': 5, 'good': 6}, 'negative': {'bad': 4, 'broken': 2}}
        self.class_total = None
        self.pr_class = None
        self.pr_class_word = None

    def fit(self, X, y, vocab):
        '''Fit training data into the model, pre-compute various counts, and probabilities based on classes and/or words, for efficiency in predict()'''

        self.vocab = vocab

        for i in range(len(y)):
            # Get class count
            self.count_class[y[i]] += 1

            for w in X[i]:
                # Get word count for each class
                # print(f'w = {w}')

                self.count_class_word[y[i]][w] += 1

        self.pr_class = {
            yi: self.count_class[yi] / len(y) if len(y) != 0 else 0 for yi in self.count_class.keys()}

        class_total = {}  # defaultdict(int)

        for yi in self.count_class.keys():
            class_total[yi] = sum(self.count_class_word[yi].values())

        self.class_total = class_total

        pr_class_word = {}
        for yi in self.count_class.keys():
            pr_class_word[yi] = {}
            for w in self.count_class_word[yi].keys():
                pr_class_word[yi][w] = self._pr_class_word_calc(yi, w)

        self.pr_class_word = pr_class_word

        # Summary info
        print(f'\nvocab: {len(self.vocab)} pr_class: {self.pr_class}')
        # print(f'pr_class_word: {self.pr_class_word}')

    def _pr_class_word_calc(self, yi, w):
        '''Calculate probability of word wi for given class yi'''

        # Handle missing word w properly: Laplace smoothing
        count = self.count_class_word[yi].get(w, 0) + self.alpha
        total = self.class_total[yi] + self.alpha * len(self.vocab)
        return count / total if total != 0 else 0

    def _log_pr(self, Xi):
        log_prs = {}

        for yi in self.pr_class.keys():
            ps = []

            for w in Xi:
                p = self._pr_class_word_calc(yi, w)
                ps.append(p)

                if p == 0:
                    break  # No need to continue

            if self.class_total[yi] != 0 and 0 not in ps:
                log_prs[yi] = np.log(self.pr_class[yi]) + np.sum(np.log(ps))
            else:  # Case of 0 probability
                log_prs[yi] = -np.inf
 
        return log_prs

    def _predict(self, Xi):
        '''Predict the class (label) for Xi (document) instance'''

        if self.use_log_pr:
            # prs = {yi: np.log(self.pr_class[yi]) + np.sum(np.log(
            #    [self._pr_class_word_calc(yi, w) for w in Xi])) for yi in self.pr_class.keys()}
            prs = self._log_pr(Xi)
        else:
            prs = {yi: self.pr_class[yi] * np.prod(
                [self._pr_class_word_calc(yi, w) for w in Xi]) for yi in self.pr_class.keys()}

        idx = np.argmax(list(prs.values()))
        ys = list(self.pr_class.keys())
        return ys[idx]    # predicted class (label)

    def predict(self, X):
        return [self._predict(Xi) for Xi in X]

    def evaluate(self, y_predict, y_test):
        diff = np.array(y_predict) == np.array(y_test)
        acc = np.mean(diff)

        TP, FP, FN, TN = 0, 0, 0, 0
        for i in range(len(diff)):
            if y_test[i] == 'positive':
                if y_predict[i] == 'positive':
                    TP += 1
                else:
                    FN += 1
            else:
                if y_predict[i] == 'positive':
                    FP += 1
                else:
                    TN += 1

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        cmat = pd.DataFrame({'positive': [TP, FN], 'negative': [FP, TN]}, index=['positive', 'negative'])

        result = {'accuracy': acc, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN, 'precision': precision,
                  'recall': recall, 'f1': {f1}, 'confusion_matrix': cmat}

        print(f'\nTP: {TP} FP: {FP} TN: {TN} FN: {FN}')
        print(f'Accuracy: {acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f} f1: {f1:.4f}')
        print(f'Confusion Matrix: \n{cmat}')

        return result


def run_mnb(train_pct_pos, train_pct_neg, test_pct_pos, test_pct_neg, alpha, use_log_pr, random_state=None):

    # Set random seed for repeatability
    if random_state is not None:
        random.seed(random_state)     # Effects picking data from csv files
        np.random.seed(random_state)  # Effects shuffling order

    # Load data
    X_train_pos, X_train_neg, vocab = load_training_set(train_pct_pos, train_pct_neg)
    X_test_pos, X_test_neg = load_test_set(test_pct_pos, test_pct_neg)

    # Prepare data
    X_train = X_train_pos + X_train_neg
    y_train = ['positive'] * len(X_train_pos) + ['negative'] * len(X_train_neg)
    X_test = X_test_pos + X_test_neg
    y_test = ['positive'] * len(X_test_pos) + ['negative'] * len(X_test_neg)

    # Shuffle data with repeatability
    X_train, y_train = shuffle(X_train, y_train, random_state=random_state)
    X_test, y_test = shuffle(X_test, y_test, random_state=random_state)


    # Train mnb model and evaluate
    mnb = MultiNB(alpha=alpha, use_log_pr=use_log_pr)
    mnb.fit(X_train, y_train, vocab)

    # print(f'mnb:\n')
    # for k, v in mnb.__dict__.items():
    #    print(f'\n{k}: {v}\n')

    y_predict = mnb.predict(X_test)

    # Peek partial results
    # k = 50
    # print(f'y_test[:{k}]: \n{y_test[:k]}')
    # print(f'y_predict[:{k}]: \n{y_predict[:k]}')

    result = mnb.evaluate(y_predict, y_test)

    return mnb, result


def alpha_impact(random_state=None):
    
    alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    
    results = []

    for alpha in alphas:
        print(f'\n----- alpha {alpha} -----')
        _, result = run_mnb(0.2, 0.2, 0.2, 0.2, alpha, True, random_state=random_state)
        results.append(result)
    
    df = pd.DataFrame(index=alphas)
    df.index.name = 'alpha'

    df['accuracy'] = [d['accuracy'] for d in results]
    df['precision'] = [d['precision'] for d in results]
    df['recall'] = [d['recall'] for d in results]

    print(df)

    fig = df['accuracy'].plot(ylabel="Accuracy", logx=True, grid=True).get_figure()
    fig.savefig('hw2_alpha_acc.png')
    fig = df.plot(logx=True, grid=True).get_figure()
    fig.savefig('hw2_alpha.png')

    return df


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_pct_pos', type=float, default=0.3,
                        help='Train percentage positive, default 0.3')
    parser.add_argument('--train_pct_neg', type=float, default=0.3,
                        help='Train percentage negative, default 0.3')
    parser.add_argument('--test_pct_pos', type=float, default=0.3,
                        help='Test percentage positive, default 0.3')
    parser.add_argument('--test_pct_neg', type=float, default=0.3,
                        help='Test percentage negative, default 0.3')
    parser.add_argument('-alpha', '--alpha', type=float,
                        default=0.0, help='alpha for smoothing, default 0.0')
    parser.add_argument('-use_log', '--use_log_pr', type=lambda s: s.lower() in [
                            'true', 'y', 'yes', '1'], default=True, help='Use log probability,default True.')
    parser.add_argument('--random_state', type=lambda s: int(s) if s.lstrip('-').strip(
    ).isdigit() else None, default=None, help='random seed like 42, -5, default None')

    args, _ = parser.parse_known_args()

    # print(f'args: {args}')

    return args


if __name__ == "__main__":
    args = get_args()

    mnb, result = run_mnb(args.train_pct_pos, args.train_pct_neg,
            args.test_pct_pos, args.test_pct_neg, args.alpha, args.use_log_pr, args.random_state)

