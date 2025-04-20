import collections
import numpy
from models import decision_tree
from models import disorder_functions
import random

class RandomForest:
    trees: list[decision_tree.DecisionTree]
    def __init__(
        self,
        ntree: int,
        minimal_size_for_split: int,
        data: numpy.ndarray,
        column_names: list[str],
        types: list[str],
        key_to_string: list[str],
        random_generator: random.Random = random.Random(),
        ):
        self.trees = []
        for _ in range(ntree):
            n_rows = data.shape[0]
            indices = numpy.random.randint(0, n_rows, (n_rows,))
            bootstrap_data = data[indices, :]
            self.trees.append(decision_tree.DecisionTree(
                bootstrap_data,
                column_names,
                disorder_functions.entropy,
                types,
                key_to_string,
                minimal_size_for_split,
                random_generator
            ))
    
    def predict(self, data: numpy.ndarray):
        if data.shape[1] != len(self.trees[0].column_names):
            raise ValueError(f"Received data with {data.shape[1]} columns. Expected {len(self.trees[0].column_names)}")
        n_rows = data.shape[0]
        # The k-th column is the k-th tree's predictions
        predictions = numpy.concatenate([
            tree.predict(data)[:, numpy.newaxis] for tree in self.trees
        ], axis=1)

        return numpy.array([
            collections.Counter(predictions[row, :]).most_common(1)[0][0]
            for row in range(n_rows)
        ], dtype=int)



