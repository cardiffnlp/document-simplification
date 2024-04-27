import json
import argparse

import sys
sys.path.append("/Users/mmaddela3/Documents/simplification_evaluation/external_repos/QuestEval")

from metrics.questeval import QuestEvalMetric

## Assumption: all metrics are in the range 0 to 100. 

def compute_metrics(dataset, metric):
    for metric in metrics:

        unique_inputs = set()
        for instance in dataset:
            org = instance["original"]
            simp1 = instance["simplification1"]
            simp2 = instance["simplification2"]
            refs = instance["references"]

            key = (org, simp1, tuple(refs))
            unique_inputs.add(key)            
            key = (org, simp2, tuple(refs))
            unique_inputs.add(key)

        unique_inputs = list(unique_inputs)
        complex = [input[0] for input in unique_inputs]
        simplified = [ input[1] for input in unique_inputs]
        references = [input[2] for input in unique_inputs]

        mvals = metric.compute_metric(complex, simplified, references)
        mvals_dict = {input: mval for input, mval in zip(unique_inputs, mvals)}

        for instance in dataset:
            org = instance["original"]
            simp1 = instance["simplification1"]
            simp2 = instance["simplification2"]
            refs = instance["references"]

            key1 = (org, simp1, tuple(refs))
            key2 = (org, simp2, tuple(refs))         
   
            instance.setdefault("metrics", {})
            instance["metrics"].setdefault(metric.name, {})
            instance["metrics"][metric.name]= {
                "simplification1": mvals_dict[key1],
                "simplification2": mvals_dict[key2],
            }

            
def pairwise_kendall(dataset):
    agg_ratings_dimension = {}

    for instance in dataset:

        for rating in instance["ratings"]:
            name = rating["name"]
            agg_ratings_dimension.setdefault(name, {})

            for metric, mvals in instance["metrics"].items():
                agg_ratings_dimension[name].setdefault(metric, {
                    "concordant": 0, "discordant": 0
                })

                pairwise_mval = 0
                mval_diff = abs(mvals["simplification2"] - mvals["simplification1"])

                # We remove comparisions with small difference in metric values.
                if mvals["simplification2"] > mvals["simplification1"] and mval_diff > 0.1:
                    pairwise_mval = 1
                human = rating["agg_value"]
                
                if pairwise_mval == human:
                    agg_ratings_dimension[name][metric]["concordant"] += 1
                else:
                    agg_ratings_dimension[name][metric]["discordant"] += 1
    
    return agg_ratings_dimension


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset")
    args=parser.parse_args()

    with open(args.dataset) as fp:
        dataset = [json.loads(line.strip()) for line in fp]

        metrics = [QuestEvalMetric()]
        compute_metrics(dataset, metrics)

        correlation_values = pairwise_kendall(dataset)
        for dimen in correlation_values:
            print(dimen)
            for metric, counts in correlation_values[dimen].items():
                print(metric)
                concordant = counts["concordant"]
                discordant = counts["discordant"]
                if (concordant + discordant) > 0:
                    tau = (concordant - discordant) / (concordant + discordant)
                    print(tau, concordant, discordant)

    with open("results_v4.json", 'w') as fp:
        json.dump(dataset, fp)

