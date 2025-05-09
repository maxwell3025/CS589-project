import csv
import datetime
import numpy
import utils
from datasets import mnist
from models import random_forest
import multiprocessing

ntree_values = [1, 2, 5, 10, 20, 50, 100]
split_sizes = [2, 5, 10, 20, 30, 40, 50]
results_file_lock = multiprocessing.Lock()
def run_eval(results_path: str, minimal_size_for_split: int, ntree: int,
             train_features: numpy.ndarray, train_labels: numpy.ndarray,
             test_features: numpy.ndarray, test_labels: numpy.ndarray):
    train_features = train_features.copy()
    train_labels = train_labels.copy()
    test_features = test_features.copy()
    test_labels = test_labels.copy()
    data = numpy.concatenate([
        train_features,
        train_labels[:, numpy.newaxis]
    ], axis = 1)
    model = random_forest.RandomForest(
        ntree = ntree,
        minimal_size_for_split = minimal_size_for_split,
        data = data,
        column_names = [str(i) for i in range(64)],
        types = ["numeric" for i in range(64)],
        key_to_string = []
    )

    test_predictions = model.predict(test_features)

    test_confusion_matrix = utils.get_confusion_matrix(test_predictions, test_labels)

    test_accuracy = utils.get_accuracy(test_confusion_matrix)
    test_f1 = utils.get_f1(test_confusion_matrix)

    results_file_lock.acquire()
    with open(results_path, "a") as file:
        writer = csv.writer(file)
        writer.writerow([f"{minimal_size_for_split=}", ntree, test_accuracy, *test_f1])
    results_file_lock.release()


results_path = f"maxwell/data/mnist_random_forest_{datetime.datetime.now().isoformat()}.csv"

with open(results_path, "a") as file:
    writer = csv.writer(file)
    writer.writerow(["hyperparameters", "ntree", "accuracy", *(f"f1_{i}" for i in range(10))])

processing_pool = multiprocessing.Pool()

processing_pool.starmap(run_eval, [(results_path, minimal_size_for_split, ntree,
                                    train_features.copy(), train_labels.copy(),
                                    test_features.copy(), test_labels.copy())
                                   for minimal_size_for_split in split_sizes
                                   for ntree in ntree_values
                                   for train_features, train_labels, test_features, test_labels
                                   in mnist.MNIST()
                                   for _ in range(1)], chunksize = 1)
