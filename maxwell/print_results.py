import pandas
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--data",  "-d", required = True, help = "The path to the .csv file whose "
                                                                "data will be graphed.")
argparser.add_argument("--opt",   "-o", default = "accuracy", help = "The column that will be optimized "
                                                                "axis for every graph.")
argparser.add_argument("--param", "-p", required = True, help = "The \"main\" column that we will "
                                                                "eventually graph.")
args = argparser.parse_args()

results = pandas.read_csv(args.data)

results.reset_index()

mean = results.groupby(["hyperparameters", args.param]).mean()
std = results.groupby(["hyperparameters", args.param]).std()

ideal_indices = mean[args.opt].groupby("hyperparameters").idxmax()
best_values = mean.join(std, lsuffix = "_mean", rsuffix = "_std").loc[ideal_indices]
best_values = best_values.sort_values(by = ["accuracy_mean"], ascending = False)

pandas.set_option('display.max_columns', None)
print(best_values)
