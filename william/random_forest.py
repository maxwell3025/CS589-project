import numpy as np
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split

from ml_utils import stratified_kfold_indices, calculate_metrics
from decision_tree import DecisionTree, most_frequent
import multiprocessing as mp


class RandomForest:
    def __init__(self, ntrees, k_fold=5, random_state=None, args=None):
        self.ntrees = ntrees
        self.ensemble = []
        self.k_fold = k_fold
        self.random_state = random_state
        self.args = args

    def predict(self, X_test):
        y_preds = []
        for dt in self.ensemble:
            y_pred = dt.predict(X_test)
            y_preds.append(y_pred)

        y_preds = np.array(y_preds)

        bagging_preds = []
        for j in range(y_preds.shape[1]):
            label = most_frequent(
                y_preds[:, j], random_state=self.random_state)
            bagging_preds.append(label)

        print(f"\nEnsemble prediction: \n{np.array(bagging_preds)}\n")

        return np.array(bagging_preds)

    def _bootstrap(self, df):
        indices = np.random.choice(df.index, size=len(df), replace=True)

        return df.loc[indices]
        # return df.sample(n=len(df), replace=True)

    def _feature_select(self, X, m_features=5):
        # No duplicate features
        features = list(X.columns)
        m_features = np.random.choice(features, size=m_features, replace=False)

        return X[list(m_features)]

    def cross_validation(self, df, random_state=None):
        
        if random_state is not None:
            np.random.seed(random_state)

        m_features = round(np.sqrt(len(df.columns[:-1])))
        y = df.iloc[:, -1]
        accuracies = []
        for train_indices, test_indices in stratified_kfold_indices(y, self.k_fold):

            for i in range(self.ntrees):
                train = df.loc[train_indices]
                # print(f"Training:\n{train}")
                test = df.loc[test_indices]
                X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

                train = self._bootstrap(train)
                X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
                print(f"Training: \n{X_train}")

                X_train = self._feature_select(
                    X_train, m_features=m_features)  # Subset of features

                tree_seed = np.random.default_rng(self.random_state).integers(1e8)

                dt = DecisionTree(min_depth=self.args.min_depth, max_depth=self.args.max_depth, min_samples=self.args.min_samples, min_info_gain=self.args.min_info_gain, random_state=tree_seed, isBST=self.args.isBST, use_gini=self.args.use_gini, early_stop_threshold=self.args.early_stop_threshold)
                dt.fit(X_train, y_train, random_state=None)
                self.ensemble.append(dt)

                y_preds = self.predict(X_test)

                acc = np.mean(y_preds == y_test)
                print(f"Ensemble accuracy: {acc:.4f}")

                accuracies.append(acc)
                # results.append((np.array(y_test), y_preds))  # Pair of expected and predicted labels

        # print(f"\n\nResults:\n{results}")
        print(f"\n\nMean Accuracy: {np.mean(accuracies):.4f}\nAccuracies:\n{accuracies}")

        return accuracies

    def cross_validation_mp(self, df, random_state=None):

        if random_state is not None:
            # random.seed(random_state)
            np.random.seed(random_state)

        # m_features = int(np.sqrt(len(df.columns[:-1])))
        m_features = round(np.sqrt(len(df.columns[:-1])))
        y = df.iloc[:, -1]

        performances = []

        q_task = mp.Queue()
        q_result = mp.Queue()

        nworkers = mp.cpu_count()

        for fold_i, (train_indices, test_indices) in enumerate(stratified_kfold_indices(y, self.k_fold)):
            
            print(f'\n----- Stratified K-Fold {fold_i} -----')

            # -----  Train -----
            
            rng = np.random.default_rng(self.random_state)

            # Send tasks  to queue
            # task = (i, seed)
            [q_task.put((i, rng.integers(1e8))) for i in range(self.ntrees)]

            # One poison pill per worker to kill
            [q_task.put(None) for i in range(nworkers)]

            # Start result collector first
            c = mp.Process(target=collector, args=(q_result, ))
            c.start()

            # Start workers
            ps = [mp.Process(target=worker_train, args=(self, q_task, q_result, df,
                             train_indices, m_features, self.args)) for i in range(nworkers)]
            [p.start() for p in ps]

            # join workers
            [p.join() for p in ps]

            # Signal collector to stop because all workers are done
            q_result.put(None)

            while True:
                if not q_result.empty():
                    trained_trees = q_result.get()
                    break

            self.ensemble = list(trained_trees.values())

            # ------- Predict -----

            test = df.loc[test_indices]
            X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

            y_pred = self.predict(X_test)

            performance = calculate_metrics(y_pred, y_test)

            performances.append(performance)

        performances = np.array(performances)
        print(f'\nntrees {self.ntrees} Performances: \n{performances}')

        avg_performance = np.mean(performances, axis=0)

        print(f'\nAverage Performance: \n{avg_performance}')

        return avg_performance, performances


"""
def worker_test(q_task, q_result, df, train_indices, test_indices, m_features, args):
    test = df.loc[test_indices]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]
    pass
"""


def worker_train(rf, q_task, q_result, df, train_indices, m_features, args):
    '''Train decision tree in random forest fashion'''

    while True:
        task = q_task.get()  # mp queue blocking read
        if task is None:   # Poison pill
            break

        i, tree_seed = task
        
        if tree_seed is not None:
            np.random.seed(tree_seed)  # Important!
        
        train = df.loc[train_indices]

        train = rf._bootstrap(train)
        X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]

        X_train = rf._feature_select(
            X_train, m_features=m_features)  # Subset of features
        print(f"    Training decision tree {i}:\n{X_train}")

        dt = DecisionTree(min_depth=args.min_depth, max_depth=args.max_depth, min_samples=args.min_samples, min_info_gain=args.min_info_gain, random_state=tree_seed, isBST=args.isBST, use_gini=args.use_gini, early_stop_threshold=args.early_stop_threshold)
        dt.fit(X_train, y_train)

        # put dt for ensemble
        q_result.put({i: dt})

        print(f'    Finish training for decision tree {i}')


def collector(q_result):
    '''Collect trained decision tree to form ensemble'''

    results = {}
    while True:
        if not q_result.empty():
            result = q_result.get()   # {i: dt} pair
            if result is not None:
                results.update(result)
            else:     # Hit by poison None, so stop
                break

    q_result.put(results)


def run_rf():

    args = get_args()

    if args.num_trees > 0:
        ntree_list = [args.num_trees]  # Single value
    elif args.num_trees == 0:
        ntree_list = [1, 5, 10, 20, 30, 40, 50]
    else:
        ntree_list = [1, 5, 10, 20, 30, 40, 50, 100, 175, 300]

    results = []
    for ntrees in ntree_list:
        rf = RandomForest(ntrees=ntrees, random_state=args.random_state, args=args)

        df = pd.read_csv(args.path)

        # rf.cross_validation(df)
        avg_performance, _ = rf.cross_validation_mp(
            df, random_state=args.random_state)

        results.append(avg_performance)

    results = np.array(results)

    df = pd.DataFrame(results, columns=[
                      'accuracy', 'precision', 'recall', 'F1'], index=ntree_list)

    print(f'\nRandom Forest Performance: \n{df}')

    # Save performance plots
    title = args.path.replace('.csv', '')

    fig = df.plot(xlabel='ntrees', grid=True, title=title).get_figure()
    fig.savefig(f'img/hw3_{title}.png')

    fig = df.plot(xlabel='ntrees', y='accuracy', ylabel='accuracy',
                  grid=True, title=title).get_figure()
    fig.savefig(f'img/hw3_{title}_accuracy.png')

    fig = df.plot(xlabel='ntrees', y='precision',
                  ylabel='precision', grid=True, title=title).get_figure()
    fig.savefig(f'img/hw3_{title}_precision.png')

    fig = df.plot(xlabel='ntrees', y='recall',
                  ylabel='recall', grid=True, title=title).get_figure()
    fig.savefig(f'img/hw3_{title}_recall.png')

    fig = df.plot(xlabel='ntrees', y='F1',
                  ylabel='F1', grid=True, title=title).get_figure()
    fig.savefig(f'img/hw3_{title}_f1.png')

    return df   # performance dataframe


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default='wdbc.csv', help='csv file path, like wdbc.csv')
    parser.add_argument('-ntrees', '--num_trees', type=int,
                        default=0, help='number of trees, default 0 (loop over [1 5 10 20 30 40 50])')
    parser.add_argument('-dmin', '--min_depth', type=int,
                        default=2, help='Min depth of tree, default 2')
    parser.add_argument('-dmax', '--max_depth', type=int,
                        default=8, help='Max depth of tree, default 8')
    parser.add_argument('-smin', '--min_samples', type=int,
                        default=2, help='Min samples for split, default 2')
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

    parser.add_argument('--random_state', type=lambda s: int(s) if s.lstrip('-').strip(
    ).isdigit() else None, default=None, help='random seed like 42, -5, default None')

    args, _ = parser.parse_known_args()

    # print(f'args: {args}')

    return args


if __name__ == "__main__":
    run_rf()
