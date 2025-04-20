import numpy
from datasets import mnist
from models import knn_predictor

dataset = mnist.MNIST()

for train_features, train_labels, test_features, test_labels in dataset:
    data = numpy.concatenate([
        train_features,
        train_labels[:, numpy.newaxis]
    ], axis=1)
    model = knn_predictor.KnnPredictor(data, 1000)

    predictions = model.predict(test_features, normalize=False)

    # Here, we generate the confusion matrix for the results that we attained.
    confusion_matrix = numpy.zeros((10, 10))
    for i in range(len(predictions)):
        confusion_matrix[test_labels[i], predictions[i]] += 1

    # Here we calculate various metrics based on the confusion matrix.
    accuracy = confusion_matrix.diagonal().sum() / confusion_matrix.sum()
    print(f"{accuracy=}")
    for i in range(10):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        tn = confusion_matrix.sum() - fp - fn - tp
        f1 = 2 * tp / (2 * tp + fp + fn)
        print(f"{i=}\t{f1=}")
