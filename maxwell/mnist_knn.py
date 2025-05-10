import csv
import datetime
import numpy
import utils
from datasets import mnist
from models import knn_predictor

dataset = mnist.MNIST()

results_path = f"maxwell/data/mnist_knn_{datetime.datetime.now().isoformat()}.csv"

with open(results_path, "a") as file:
    writer = csv.writer(file)
    writer.writerow(["hyperparameters", "k", "accuracy", *(f"f1_{i}" for i in range(10))])

k_values = [*range(1, 100)]
for k in k_values:
    for train_features, train_labels, test_features, test_labels in dataset:
        data = numpy.concatenate([
            train_features,
            train_labels[:, numpy.newaxis]
        ], axis=1)
        model = knn_predictor.KnnPredictor(data, k)

        test_predictions = model.predict(test_features, normalize=False)

        test_confusion_matrix = utils.get_confusion_matrix(test_predictions, test_labels)

        test_accuracy = utils.get_accuracy(test_confusion_matrix)
        test_f1 = utils.get_f1(test_confusion_matrix)

        with open(results_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow(["ignore", k, test_accuracy, *test_f1])
