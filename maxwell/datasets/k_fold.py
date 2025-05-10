import numpy
import pandas
from typing import Literal, cast

class KFold:
    k: int
    data: list[numpy.ndarray]
    key_to_str: list[str] = []
    column_names: list[str]
    column_dtypes: list[Literal["categorical", "numeric"]] = []


    def __init__(self, csv_url: str, k: int):
        self.k = k
        dataframe = pandas.read_csv(csv_url)
        dataframe.label = pandas.Categorical(dataframe.label).codes
        dataframe = dataframe.sample(frac=1)
        cat_values: set[str] = set()

        categorical_column_names = []
        numeric_column_names = []
        for column_name in dataframe.columns:
            if column_name.endswith("_cat"):
                categorical_column_names.append(column_name)
                dataframe[column_name] = dataframe[column_name].astype(str)
                cat_values = cat_values.union(dataframe[column_name].unique())
            elif column_name.endswith(("_num")):
                numeric_column_names.append(column_name)
                dataframe[column_name] = dataframe[column_name].astype(float)
        self.column_names = [*categorical_column_names, *numeric_column_names]
        dataframe = dataframe[[*categorical_column_names, *numeric_column_names, "label"]]

        self.column_dtypes = [
            *(cast(Literal["categorical"], "categorical") for _ in categorical_column_names),
            *(cast(Literal["numeric"], "numeric") for _ in numeric_column_names)
        ]
        str_to_key: dict[str, int] = {}
        key_to_str: list[str] = []
        for cat_value in cat_values:
            str_to_key[cat_value] = len(key_to_str)
            key_to_str.append(cat_value)
        self.key_to_str = key_to_str
        
        dataframe[categorical_column_names] = dataframe[categorical_column_names].map(lambda x: str_to_key[x])

        negative_dataframe = dataframe[dataframe["label"] == 0]
        positive_dataframe = dataframe[dataframe["label"] == 1]
        positive_data_tensor = positive_dataframe.to_numpy(dtype=float)
        negative_data_tensor = negative_dataframe.to_numpy(dtype=float)
        n_rows_positive = positive_data_tensor.shape[0]
        n_rows_negative = negative_data_tensor.shape[0]
        self.data = []
        for i in range(k):
            data_tensor = numpy.concatenate([
                positive_data_tensor[i * n_rows_positive // k:(i + 1) * n_rows_positive // k, :],
                negative_data_tensor[i * n_rows_negative // k:(i + 1) * n_rows_negative // k, :],
            ], axis=0)
            numpy.random.shuffle(data_tensor)
            self.data.append(data_tensor)
        

    def __iter__(self):
        for i in range(self.k):
            test_data = self.data[i]
            train_data = self.data[:i] + self.data[i + 1:]
            train_data = numpy.concat(train_data, axis = 0)
            yield test_data, train_data
