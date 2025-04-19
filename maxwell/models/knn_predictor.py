import collections
import numpy

class KnnPredictor:
    def __init__(self, data: numpy.ndarray, k: int, norm_correction = 0.01) -> None:
        self.points = data[:, :-1]
        self.labels = data[:, -1]
        self.k = k
        self.inv_normalization = 1.0 / (self.points.std(axis = 0, ddof = 1) + norm_correction)

    def predict(self, features: numpy.ndarray, normalize: bool = True):
        """ Predict a vector of features
        """
        predictions = []
        if features.shape[1] != self.points.shape[1]:
            raise Exception("Wrong shape!")

        for feature_index in range(features.shape[0]):
            feature_vector = features[feature_index:feature_index + 1, :]
            differences = self.points - feature_vector
            if normalize:
                normalized_differences = differences * self.inv_normalization[numpy.newaxis, :]
            else:
                normalized_differences = differences
            distances = (normalized_differences ** 2).sum(axis=1)
            nearest_indices = numpy.argpartition(distances, self.k)[:self.k]
            predictions.append(collections.Counter(self.labels[nearest_indices]).most_common(1)[0][0])

        return numpy.array(predictions, dtype=int)
