import csv
import datetime
import utils
from datasets import k_fold
from models import random_forest
import multiprocessing


results_file_lock = multiprocessing.Lock()
def run_eval(results_path: str):
    global results_file_lock
    dataset = k_fold.KFold("dataset/rice.csv", k = 10)

    ntree_values = [1, 2, 5, 10, 20, 50, 100]
    for ntree in ntree_values:
        for test, train in dataset:
            model = random_forest.RandomForest(
                ntree=ntree,
                minimal_size_for_split = 10,
                data=train,
                column_names=dataset.column_names,
                types=dataset.column_dtypes,
                key_to_string=dataset.key_to_str
            )

            test_predictions = model.predict(test[:, :-1])

            test_confusion_matrix = utils.get_confusion_matrix(test_predictions, test[:, -1])

            test_accuracy = utils.get_accuracy(test_confusion_matrix)
            test_f1 = utils.get_f1(test_confusion_matrix)

            results_file_lock.acquire()
            with open(results_path, "a") as results_file:
                results_writer = csv.writer(results_file)
                results_writer.writerow([ntree, test_accuracy, *test_f1])
            results_file_lock.release()


results_path = f"maxwell/data/rice_random_forest_{datetime.datetime.now().isoformat()}.csv"

with open(results_path, "a") as results_file:
    writer = csv.writer(results_file)
    writer.writerow(["hyperparameters", "ntree", "accuracy", *(f"f1_{i}" for i in range(2))])

processing_pool = multiprocessing.Pool()

processing_pool.map(run_eval, [results_path] * 16)
