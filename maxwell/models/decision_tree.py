import fast_algorithms
import numpy
import random
import math
from typing import Callable, Union


class DecisionTree:
    children: Union[dict[int, "DecisionTree"], None]
    discriminant: int
    parent: Union["DecisionTree", None]
    prediction: Union[int, None]
    column_names: list[str]
    key_to_string: list[str]
    discriminant_dtype: str
    discriminant_threshold: float

    def __init__(
        self,
        data: numpy.ndarray,
        column_names: list[str],
        disorder_function: Callable[[numpy.ndarray], float],
        types: list[str],
        key_to_string: list[str],
        minimal_size_for_split: int,
        random_generator: random.Random,
        splittable_set: set[int] | None = None,
        ) -> None:
        n_cols = data.shape[1]
        n_rows = data.shape[0]
        n_features = n_cols - 1

        if splittable_set == None:
            splittable_set = {*range(0, n_features)}

        if len(types) != n_features:
            raise ValueError(f"Data contains {n_features} features. Recieved {len(types)} column types.")
        if len(column_names) != n_features:
            raise ValueError(f"Data contains {n_features} features. Recieved {len(column_names)} column names.")

        # my_splittable_set = {x for x in splittable_set if random_generator.random() < math.sqrt(1 / n_cols)}
        my_splittable_set = numpy.random.choice(
            list(splittable_set),
            (math.ceil(math.sqrt(len(splittable_set))),),
            False
        )

        self.parent = None
        self.column_names = column_names
        self.key_to_string = key_to_string

        label_values, label_counts = fast_algorithms.unique_counts(data[:, -1])
        self.prediction = label_values[label_counts.argmax()]

        if len(my_splittable_set) == 0 or n_rows < minimal_size_for_split:
            self.children = None
            return

        # Calculate the resulting disorder for each partition
        total_disorder: list[float] = [0] * (n_features)
        for discriminant_index in range(n_features):
            if discriminant_index not in my_splittable_set:
                total_disorder[discriminant_index] = float("inf")
                continue
            if types[discriminant_index] == "categorical":
                label_values = fast_algorithms.unique(data[:, discriminant_index])
                for value in label_values:
                    partition = data[data[:, discriminant_index] == value]
                    total_disorder[discriminant_index] += disorder_function(partition) * partition.shape[0]
            elif types[discriminant_index] == "numeric":
                threshold = numpy.mean(data[:, discriminant_index])
                high_partition = data[data[:, discriminant_index] > threshold]
                low_partition = data[data[:, discriminant_index] <= threshold]
                total_disorder[discriminant_index] += disorder_function(high_partition) * high_partition.shape[0]
                total_disorder[discriminant_index] += disorder_function(low_partition) * low_partition.shape[0]
        
        self.discriminant = int(numpy.argmin(total_disorder))
        self.discriminant_dtype = types[self.discriminant]

        if self.discriminant_dtype == "numeric":
            self.discriminant_threshold = numpy.mean(data[:, self.discriminant]).item()
            self.children = dict()
            relevant_rows = data[data[:, self.discriminant] <= self.discriminant_threshold]
            if relevant_rows.shape[0]:
                self.children[0] = DecisionTree(
                    relevant_rows,
                    self.column_names,
                    disorder_function,
                    types,
                    key_to_string,
                    minimal_size_for_split,
                    random_generator,
                    {column for column in splittable_set if column != self.discriminant},
                )
                self.children[0].parent = self

            relevant_rows = data[data[:, self.discriminant] > self.discriminant_threshold]
            if relevant_rows.shape[0]:
                self.children[1] = DecisionTree(
                    relevant_rows,
                    self.column_names,
                    disorder_function,
                    types,
                    key_to_string,
                    minimal_size_for_split,
                    random_generator,
                    {column for column in splittable_set if column != self.discriminant}
                )
                self.children[1].parent = self
        elif self.discriminant_dtype == "categorical":
            label_values = fast_algorithms.unique(data[:, self.discriminant])
            self.children = dict()
            for value in label_values:
                relevant_rows = data[data[:, self.discriminant] == value]
                if relevant_rows.shape[0]:
                    self.children[value] = DecisionTree(
                        relevant_rows,
                        self.column_names,
                        disorder_function,
                        types,
                        key_to_string,
                        minimal_size_for_split,
                        random_generator,
                        {column for column in splittable_set if column != self.discriminant}
                    )
                    self.children[value].parent = self
        else:
            raise RuntimeError("no discriminant dtype")

    def predict(self, data: numpy.ndarray) -> numpy.ndarray:
        if self.children == None:
            return numpy.full((data.shape[0],), self.prediction, dtype = numpy.float64)
        
        predictions = numpy.full((data.shape[0],), 0.5, dtype = numpy.float64)
        for discriminant_value in self.children:
            if self.discriminant_dtype == "categorical":
                indices = numpy.argwhere(data[:, self.discriminant] == discriminant_value)[:, 0]
            elif self.discriminant_dtype == "numeric" and discriminant_value == 0:
                indices = numpy.argwhere(data[:, self.discriminant] <= self.discriminant_threshold)[:, 0]
            elif self.discriminant_dtype == "numeric" and discriminant_value == 1:
                indices = numpy.argwhere(data[:, self.discriminant] > self.discriminant_threshold)[:, 0]
            else:
                raise RuntimeError("Failed to evaluate node type")
            if indices.size:
                if discriminant_value in self.children:
                    predictions[indices] = self.children[discriminant_value].predict(data[indices, :])
                else:
                    predictions[indices] = numpy.full((indices.size,), self.prediction, dtype = numpy.float64)
        return predictions

    @property
    def is_root(self):
        return self.parent == None

    @property
    def is_leaf(self):
        return self.children == None

    def _str(self, lines: list[str], label: str, line_stack: list[str], is_last: bool) -> None:
        LINE = "\u2502"
        SIDE_TEE = "\u251c"
        TEE = "\u252c"
        BEND = "\u2514"
        BAR = "\u2500"
        LEQ = "\u2264"
        if len(line_stack) == 0:
            lines.append(f"{label}")
        else:
            base = "".join(line_stack[:-1]) + (BEND if is_last else SIDE_TEE) + BAR + label
            if self.is_leaf:
                base = f"{base}: {self.prediction}" # type: ignore
            lines.append(base)
        if not self.is_leaf:
            for child_index, child_value in enumerate(self.children.keys()): # type: ignore
                new_line_stack = line_stack + [" ", LINE]
                if is_last and len(line_stack) > 0:
                    new_line_stack[-3] = " "
                is_last_child = child_index == len(self.children) - 1 # type: ignore

                if self.discriminant_dtype == "categorical":
                    label = f"{self.column_names[self.discriminant]} = {self.key_to_string[child_value]}"
                elif self.discriminant_dtype == "numeric" and child_value == 0:
                    label = f"{self.column_names[self.discriminant]} {LEQ} {self.discriminant_threshold}"
                elif self.discriminant_dtype == "numeric" and child_value == 1:
                    label = f"{self.column_names[self.discriminant]} > {self.discriminant_threshold}"
                self.children[child_value]._str(lines, label, new_line_stack, is_last_child) # type: ignore
    
    def __str__(self) -> str:
        lines: list[str] = []
        self._str(lines, " .", [], True)
        return "\n".join(lines) + "\n"
