import csv
from datasets import k_fold
import datetime
from models import annotated_tensor
from models import multi_layer_perceptron
import numpy
import utils

def run_eval(results_path: str):
    n_epochs = 1000
    learning_rate = 0.1
    regularization_cost = 0.01

    dataset = k_fold.KFold("dataset/rice.csv", k = 10)

    model_shapes = [[1, 1],
                    [1, 2,  1],
                    [1, 4,  1],
                    [1, 8,  1],
                    [1, 16, 1],
                    [1, 2,  2,  1],
                    [1, 4,  2,  1],
                    [1, 8,  2,  1],
                    [1, 16, 2,  1],
                    [1, 2,  4,  1],
                    [1, 4,  4,  1],
                    [1, 8,  4,  1],
                    [1, 16, 4,  1],
                    [1, 2,  8,  1],
                    [1, 4,  8,  1],
                    [1, 8,  8,  1],
                    [1, 16, 8,  1],
                    [1, 2,  16, 1],
                    [1, 4,  16, 1],
                    [1, 8,  16, 1],
                    [1, 16, 16, 1]]

    for model_shape in model_shapes:
        print(model_shape)
        for test, train in dataset:
            train_features = train[:, :-1]
            train_labels = train[:, -1]

            test_features = test[:, :-1]
            test_labels = test[:, -1]

            norm_bias = numpy.mean(train_features, axis = 0)
            norm_scale = numpy.std(train_features, axis = 0, ddof = 1)

            train_features = train_features - norm_bias[numpy.newaxis, :]
            train_features = train_features / norm_scale[numpy.newaxis, :]

            test_features = test_features - norm_bias[numpy.newaxis, :]
            test_features = test_features / norm_scale[numpy.newaxis, :]
            
            input_size = train_features.shape[1]
            model_shape[0] = input_size
            model = multi_layer_perceptron.MultiLayerPerceptron(model_shape)

            for epoch_num in range(n_epochs):
                prediction = model.predict(train_features)
                cost = annotated_tensor.cross_entropy(prediction, train_labels[numpy.newaxis, :])
                cost = annotated_tensor.sum(cost)
                cost = cost * (1 / train.shape[0])
                cost = cost + model.reg_cost() * regularization_cost

                cost.add_gradient(numpy.array([[-1]]))
                model.step(learning_rate)
                model.zero_grad()

                test_predictions = model.predict(test_features).value.squeeze()

                test_predictions = (test_predictions > 0.5).astype(int, casting = "safe")

                test_confusion_matrix = utils.get_confusion_matrix(test_predictions, test_labels)

                test_accuracy = utils.get_accuracy(test_confusion_matrix)
                test_f1 = utils.get_f1(test_confusion_matrix)

                with open(results_path, "a") as results_file:
                    results_writer = csv.writer(results_file)
                    results_writer.writerow([str(model_shape), epoch_num, test_accuracy, *test_f1])


results_path = f"maxwell/data/rice_nn_{datetime.datetime.now().isoformat()}.csv"

with open(results_path, "a") as results_file:
    writer = csv.writer(results_file)
    writer.writerow(["hyperparameters", "epoch", "accuracy", *(f"f1_{i}" for i in range(2))])

for _ in range(1):
    run_eval(results_path)
