import math
import numpy

def full_log(x: float):
    return float("-inf") if x == 0 else math.log(x)

class MultiBayesModel:
    initial_evidence: numpy.ndarray
    default_evidence: numpy.ndarray
    evidence: dict[str, numpy.ndarray]

    def __init__(self, data: list[list[list[str]]], vocab: set[str], laplace_smoothing_coeff = 0) -> None:
        self.evidence = {term: numpy.zeros((len(data),), numpy.float64) for term in vocab}
        self.initial_evidence = numpy.zeros((len(data),), numpy.float64)

        total_articles = 0

        for class_index, class_data in enumerate(data):
            total_articles += len(class_data)
            total_size = laplace_smoothing_coeff * len(vocab)
            frequencies = {term: laplace_smoothing_coeff for term in vocab}
            for article in class_data:
                for term in article:
                    frequencies[term] += 1
                total_size += len(article)
            for term in frequencies:
                self.evidence[term][class_index] = full_log((frequencies[term])/total_size)

        for class_index, class_data in enumerate(data):
            self.initial_evidence[class_index] = full_log((len(class_data))/total_articles)

        self.default_evidence = numpy.full((len(data),), full_log(laplace_smoothing_coeff/total_articles))

    
    def predict(self, article: list[str]) -> int:
        # When predicting, we assume we only get one sample per word.
        evidence = numpy.copy(self.initial_evidence)
        for term in set(article):
            if term in self.evidence:
                evidence = evidence + self.evidence[term]
            else:
                evidence = evidence + self.default_evidence
        max_evidence: float = evidence.max()
        # Implementation based on https://campuswire.com/c/G5E8C25E9/feed/115
        return numpy.random.choice(numpy.argwhere(evidence == max_evidence).squeeze(1))

    def __str__(self):
        return f"{", ".join(map(lambda value: f"{value}", self.initial_evidence))}"
