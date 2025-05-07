import numpy
import typing

def get_confusion_matrix(predictions: numpy.ndarray, labels: numpy.ndarray) -> numpy.ndarray:
    n_classes = max(predictions.max(), labels.max()) + 1
    confusion_matrix = numpy.zeros((n_classes, 10))
    for i in range(len(predictions)):
        confusion_matrix[int(labels[i]), int(predictions[i])] += 1
    return confusion_matrix

def get_accuracy(confusion_matrix: numpy.ndarray):
    return confusion_matrix.diagonal().sum() / confusion_matrix.sum()

@typing.overload
def get_f1(confusion_matrix: numpy.ndarray, class_index: int) -> float:
    ...

@typing.overload
def get_f1(confusion_matrix: numpy.ndarray) -> list[float]:
    ...

def get_f1(confusion_matrix: numpy.ndarray, class_index: int | None = None):
    if class_index == None:
        return [
            get_f1(confusion_matrix, i)
            for i in range(confusion_matrix.shape[0])
        ]
    tp = confusion_matrix[class_index, class_index]
    fp = confusion_matrix[:, class_index].sum() - tp
    fn = confusion_matrix[class_index, :].sum() - tp
    # tn = confusion_matrix.sum() - fp - fn - tp
    f1 = 2 * tp / (2 * tp + fp + fn)
    return f1
