import csv
import datetime
import numpy
import utils
from datasets import mnist
from models import random_forest

dataset = mnist.MNIST()

results_path = f"maxwell/data/mnist_random_forest_{datetime.datetime.now().isoformat()}.csv"

with open(results_path, "a") as file:
    writer = csv.writer(file)
    writer.writerow(["ntree", "accuracy", *(f"f1_{i}" for i in range(10))])

ntree_values = [1, 2, 5, 10, 20, 50, 100]
for ntree in ntree_values:
    for train_features, train_labels, test_features, test_labels in dataset:
        data = numpy.concatenate([
            train_features,
            train_labels[:, numpy.newaxis]
        ], axis=1)
        model = random_forest.RandomForest(
            ntree=ntree,
            minimal_size_for_split=10,
            data=data,
            column_names=[str(i) for i in range(64)],
            types=["numeric" for i in range(64)],
            key_to_string=[]
        )

        test_predictions = model.predict(test_features)

        test_confusion_matrix = utils.get_confusion_matrix(test_predictions, test_labels)

        test_accuracy = utils.get_accuracy(test_confusion_matrix)
        test_f1 = utils.get_f1(test_confusion_matrix)

        with open(results_path, "a") as file:
            writer = csv.writer(file)
            writer.writerow([ntree, test_accuracy, *test_f1])
