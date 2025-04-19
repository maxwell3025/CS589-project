import numpy
import decision_tree
import disorder_functions
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
        random_generator: random.Random,
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
    
    def __call__(self, data: numpy.ndarray):
        if data.shape[1] != len(self.trees[0].column_names):
            raise ValueError(f"Received data with {data.shape[1]} columns. Expected {len(self.trees[0].column_names)}")
        # NOTE this only works with binary data!
        prediction_sum = numpy.zeros((data.shape[0]))
        for tree in self.trees:
            prediction_sum += tree.predict(data)
        prediction = prediction_sum / len(self.trees)
        return numpy.round(prediction)



