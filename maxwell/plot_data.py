import pandas
from matplotlib import pyplot
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--data",  required = True, help = "The path to the .csv file whose data "
                                                          "will be graphed.")
argparser.add_argument("--index", required = True, help = "The column that will be the x axis for "
                                                          "every graph.")
argparser.add_argument("--name",  required = True, help = "A prefix for all of the generated file "
                                                          "names.")
 
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