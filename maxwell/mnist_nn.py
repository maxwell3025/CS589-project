import csv
from datasets import k_fold
from datasets import mnist
import datetime
from models import annotated_tensor
from models import multi_layer_perceptron
import numpy
import utils

def run_eval(results_path: str):
    n_epochs = 1000
    learning_rate = 0.1
    regularization_cost = 0.01

    dataset = mnist.MNIST()

    model_shapes = [[1, 10],
                    [1, 8,  10],
                    [1, 16, 10],
                    [1, 64, 10],
                    [1, 128, 10],
                    [1, 16, 16, 10],
                    [1, 8,  8,  10]]

    for model_shape in model_shapes:
        for train_features, train_labels, test_features, test_labels in dataset:
            train_features = train_features / 16
            test_features = test_features / 16

            input_size = train_features.shape[1]
            model_shape[0] = input_size
            model = multi_layer_perceptron.MultiLayerPerceptron(model_shape)
            print(model_shape)

            for epoch_num in range(n_epochs):
                prediction = model.predict(train_features)
                cost = annotated_tensor.cross_entropy(prediction, numpy.eye(10)[train_labels].transpose())
                cost = annotated_tensor.sum(cost)
                cost = cost * (1 / train_features.shape[0])
                cost = cost + model.reg_cost() * regularization_cost

                cost.add_gradient(numpy.array([[-1]]))
                model.step(learning_rate)
                model.zero_grad()

                test_predictions = model.predict(test_features).value.squeeze()

                test_predictions = test_predictions.argmax(axis = 0)

                test_confusion_matrix = utils.get_confusion_matrix(test_predictions, test_labels)

                test_accuracy = utils.get_accuracy(test_confusion_matrix)
                test_f1 = utils.get_f1(test_confusion_matrix)

                with open(results_path, "a") as results_file:
                    results_writer = csv.writer(results_file)
                    results_writer.writerow([f"{model_shape=}", epoch_num, test_accuracy, *test_f1])


results_path = f"maxwell/data/mnist_nn_{datetime.datetime.now().isoformat()}.csv"

with open(results_path, "a") as results_file:
    writer = csv.writer(results_file)
    writer.writerow(["hyperparameters", "epoch", "accuracy", *(f"f1_{i}" for i in range(10))])

for _ in range(1):
    run_eval(results_path)
