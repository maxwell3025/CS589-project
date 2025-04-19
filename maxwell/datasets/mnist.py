from sklearn import datasets
import numpy

class MNIST:
    features: list[numpy.ndarray]
    labels: list[numpy.ndarray]
    size: int

    def __init__(self):
        raw_features, raw_labels = datasets.load_digits(return_X_y=True)
        shuffle_indices = numpy.random.permutation(len(raw_labels))
        raw_features = raw_features[shuffle_indices, :]
        raw_labels = raw_labels[shuffle_indices]

        sorted_indices = numpy.argsort(raw_labels, stable=True)
        self.features = raw_features[sorted_indices, :]
        self.labels = raw_labels[sorted_indices]
        self.size = len(self.labels)

    def __iter__(self):
        for i in range(10):
            train_features = numpy.concatenate([
                self.features[j::10, :] for j in range(10) if j != i
            ], axis=0)

            train_labels = numpy.concatenate([
                self.labels[j::10] for j in range(10) if j != i
            ], axis=0)

            test_features = self.features[i::10, :]

            test_labels = self.labels[i::10]

            yield train_features, train_labels, test_features, test_labels
