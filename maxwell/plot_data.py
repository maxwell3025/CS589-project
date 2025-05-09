import pandas
from matplotlib import pyplot
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--data",   "-d", required = True, help = "The path to the .csv file whose "
                                                                 "data will be graphed.")
argparser.add_argument("--x-axis", "-x", required = True, help = "The column that will be the x "
                                                                 "axis for every graph.")
argparser.add_argument("--name",   "-n", required = True, help = "A prefix for all of the "
                                                                 "generated file names.")
argparser.add_argument("--filter", "-f", required = True, help = "The value that "
                                                                 "\"hyperparameter\" must match.") 
args = argparser.parse_args()

x_axis = args.x_axis

results = pandas.read_csv(args.data)
results = results[results.hyperparameter == args.filter]

for column in results.columns:
    results.groupby(x_axis).mean().plot(y=column, yerr=results.groupby(x_axis).std(), legend=False)
    pyplot.title(f"{x_axis} v.s {column}")
    pyplot.xlabel(x_axis)
    pyplot.ylabel(column)
    pyplot.grid(True)
    pyplot.savefig(f"maxwell/figures/{args.name}_{column}.pdf", bbox_inches="tight")
    pyplot.clf()
