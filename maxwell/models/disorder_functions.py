import numpy

def entropy(data: numpy.ndarray) -> float:
    if data.shape[0] == 0:
        return 0
    labels = data[:, -1]
    counts = numpy.bincount(labels.astype(numpy.uint8, casting="unsafe"))
    counts = counts.compress(counts)
    # Since Python has signed zeroes, I return positive zero on no entropy to make outputs look nice
    if counts.size == 1:
        return 0
    frequencies = counts / labels.size
    entropy = -(frequencies * numpy.log(frequencies)).sum()
    return entropy

def gini(data: numpy.ndarray) -> float:
    if data.shape[0] == 0:
        return 0
    labels = data[:, -1]
    counts = numpy.bincount(labels)
    counts = counts.compress(counts)
    # Since Python has signed zeroes, I return positive zero on no entropy to make outputs look nice
    if counts.size == 1:
        return 0
    frequencies = counts / labels.size
    return 1 - (frequencies * frequencies).sum()
