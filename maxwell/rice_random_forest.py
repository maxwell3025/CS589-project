import csv
import datetime
import numpy
import utils
from datasets import k_fold
from models import random_forest

dataset = k_fold.KFold("dataset/rice.csv", k = 5)

results_path = f"maxwell/data/rice_random_forest_{datetime.datetime.now().isoformat()}.csv"

with open(results_path, "a") as file:
    writer = csv.writer(file)
    writer.writerow(["ntree", "accuracy", *(f"f1_{i}" for i in range(2))])

ntree_values = [1, 2, 5, 10, 20, 50, 100]
for ntree in ntree_values:
    for test, train in dataset:
        model = random_forest.RandomForest(
            ntree=ntree,
            minimal_size_for_split=10,
            data=train,
            column_names=dataset.column_names,
            types=dataset.column_dtypes,
            key_to_string=dataset.key_to_str
        )

        test_predictions = model.predict(test[:, :-1])

        test_confusion_matrix = utils.get_confusion_matrix(test_predictions, test[:, -1])

        test_accuracy = utils.get_accuracy(test_confusion_matrix)
        test_f1 = utils.get_f1(test_confusion_matrix)

        with open(results_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow([ntree, test_accuracy, *test_f1])
