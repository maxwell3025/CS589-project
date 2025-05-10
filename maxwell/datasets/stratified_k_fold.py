import numpy
import pandas
from typing import Literal, Iterator

def get_dataset(path: str, k: int = 5) -> Iterator[tuple[numpy.ndarray, numpy.ndarray]]:
    dataframe = pandas.read_csv(path)

    tensors = []
    for column_name in dataframe.columns:
        if column_name.endswith("_cat"):
            categorical_data = pandas.Categorical(dataframe[column_name])
            tensor = numpy.eye(len(categorical_data.categories))[categorical_data.codes]
            tensors.append(tensor)
        elif column_name.endswith("_num"):
            tensors.append(dataframe[column_name].array[:, numpy.newaxis])
    tensors.append(dataframe["label"].array[:, numpy.newaxis])

    data = numpy.concatenate(tensors, axis = 1, dtype = float)

    positive_cases = data[numpy.argwhere(data[:, -1] == 1)[:, 0]]
    negative_cases = data[numpy.argwhere(data[:, -1] == 0)[:, 0]]

    numpy.random.shuffle(positive_cases)
    numpy.random.shuffle(negative_cases)

    folds: list[numpy.ndarray] = []
    for i in range(k):
        current_fold_positive = positive_cases[
            len(positive_cases) * i // k:len(positive_cases) * (i + 1) // k]
        current_fold_negative = negative_cases[
            len(negative_cases) * i // k:len(negative_cases) * (i + 1) // k]
        current_fold = numpy.concatenate([current_fold_positive, current_fold_negative], axis = 0)
        numpy.random.shuffle(current_fold)
        folds.append(current_fold)

    for i in range(k):
        yield (numpy.concatenate(folds[:i] + folds[i + 1:], axis = 0),
               folds[i])
