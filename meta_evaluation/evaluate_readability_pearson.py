import json
import argparse

from scipy.stats import pearsonr

from metrics.sari import SARI
from metrics.bleu import BLEU
from metrics.ari import ARI
from metrics.dalechall import DalleChall
from metrics.fkgl import FKGL
from metrics.fre import FRE
from metrics.trank import TRank


def compute_metrics(dataset, metric):
    for metric in metrics:

        unique_inputs = set()
        for instance in dataset:
            org = instance["original"]
            simp = instance["simplification"]
            refs = instance["references"]
            key = (org, simp, tuple(refs))
            unique_inputs.add(key)

        unique_inputs = list(unique_inputs)

        complex = [input[0] for input in unique_inputs]
        simplified = [ input[1] for input in unique_inputs]
        references = [input[2] for input in unique_inputs]

        mvals = metric.compute_metric(complex, simplified, references)
        mvals_dict = {input: mval for input, mval in zip(unique_inputs, mvals)}

        for instance in dataset:
            org = instance["original"]
            simp = instance["simplification"]
            refs = instance["references"]
            key = (org, simp, tuple(refs))
            instance.setdefault("metrics", {})
            instance["metrics"].setdefault(metric.name, {})
            instance["metrics"][metric.name]= mvals_dict[key]


def print_pearson(dataset, metrics):
    dimensions = [rating["name"] for rating in dataset[0]["ratings"]]
    for dimension in dimensions:
        print(dimension)
        for metric in metrics:
            mvals, human_vals = [], []
            for instance in dataset:
                mvals.append(instance["metrics"][metric.name])
                for ratings in instance['ratings']:
                    if ratings['name'] == dimension:
                        ratings_val = ratings["agg_value"]
                        break
                human_vals.append(ratings_val)
            corr, _ = pearsonr(mvals, human_vals)
            print(metric.name, corr)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    args=parser.parse_args()

    with open(args.dataset) as fp:
        dataset = [json.loads(line.strip()) for line in fp]

        metrics = [
                #    SARI(),
                #    BLEU(),
                #    ARI(),
                #    DalleChall(),
                #    FKGL(),
                #    FRE(),
                   TRank(),
        ]
        
        compute_metrics(dataset, metrics)
        print_pearson(dataset, metrics)
        with open(args.output, 'w') as fp:
            json.dump(dataset, fp)
