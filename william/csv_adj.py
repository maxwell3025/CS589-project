import numpy as np
import pandas as pd

def convert_csv(csv_path, encoding=None, suffix='_adj', verbose=True):
    print(f'\n----- {csv_path} -----\n')
    df = pd.read_csv(csv_path)

    if verbose:
        print(df)
        print(df.describe())

    uniques = {}
    for col in df.columns:
        non_numeric = {x for x in df[col] if not pd.api.types.is_number(x)}
        if non_numeric:
            uniques[col] = non_numeric

    print(f'\nNon-numeric uniques:\n{uniques}')
    print(f'Use above to implement the enencoding.\n')

    if encoding:
        for col in encoding.keys():
            df[col] = [encoding[col][x] for x in df[col]]

        if verbose:
            print(df)

        df.to_csv(f"{csv_path.replace('.csv', f'{suffix}.csv')}", index=None)


def proj_csv_convert():

    convert_csv('../dataset/credit_approval.csv', encoding=None)
    # Convert credit_approval.csv
    # Encode non-numerical values in uniques
    encoding = {'attr1_cat': {'a': 0, 'b': 1},
                'attr4_cat': {'l': 0, 'u': 1, 'y': 2},
                'attr5_cat': {'g': 0, 'gg': 1, 'p': 2},
                'attr6_cat': {'aa': 0, 'c': 1, 'cc': 2, 'd': 3,
                              'e': 4, 'ff': 5, 'i': 6, 'j': 7,
                              'k': 8, 'm': 9, 'q': 10, 'r': 11,
                              'w': 12, 'x': 13},
                'attr7_cat': {'bb': 0, 'dd': 1, 'ff': 2, 'h': 3, 
                              'j': 4, 'n': 5, 'o': 6, 'v': 7,
                              'z': 8},
                'attr9_cat': {'f': 0, 't': 1},
                'attr10_cat': {'f': 0, 't': 1},
                'attr12_cat': {'f': 0, 't': 1},
                'attr13_cat': {'g': 0, 'p': 1, 's': 2}
                }

    convert_csv('../dataset/credit_approval.csv', encoding=encoding)

if __name__ == "__main__":
    proj_csv_convert()
