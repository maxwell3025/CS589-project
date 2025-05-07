import csv
import datetime
import numpy
import utils
from datasets import k_fold
from models import knn_predictor

dataset = k_fold.KFold("dataset/rice.csv", k = 10)

results_path = f"maxwell/data/rice_knn_{datetime.datetime.now().isoformat()}.csv"

with open(results_path, "a") as file:
    writer = csv.writer(file)
    writer.writerow(["k", "accuracy", *(f"f1_{i}" for i in range(2))])

k_values = [*range(1, 100)]
for k in k_values:
    for test, train in dataset:
        model = knn_predictor.KnnPredictor(train, k)

        test_predictions = model.predict(test[:, :-1], normalize=False)

        test_confusion_matrix = utils.get_confusion_matrix(test_predictions, test[:, -1])

        test_accuracy = utils.get_accuracy(test_confusion_matrix)
        test_f1 = utils.get_f1(test_confusion_matrix)

        with open(results_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow([k, test_accuracy, *test_f1])
