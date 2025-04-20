import pandas
from matplotlib import pyplot
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--data", required=True)
argparser.add_argument("--index", required=True)
argparser.add_argument("--name", required=True)

args = argparser.parse_args()

index = args.index
name = args.name

results = pandas.read_csv(args.data, index_col=index)

for column in results.columns:
    results.groupby(index).mean().plot(y=column, yerr=results.groupby(index).std(), legend=False)
    pyplot.title(f"{index} v.s {column}")
    pyplot.xlabel(index)
    pyplot.ylabel(column)
    pyplot.grid(True)
    pyplot.savefig(f"maxwell/figures/{name}_{column}.pdf", bbox_inches="tight")
    pyplot.clf()