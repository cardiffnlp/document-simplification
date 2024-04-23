import json
import argparse

import sys

# sys.path.append('/Users/mmaddela3/Documents/simplification_evaluation/external_repos/google-research') 


from metrics.sari import SARI
from metrics.bleu import BLEU
from metrics.gleu import GLEU
from metrics.D_SARI import DSARI
from metrics.bscore import BERTScore
from metrics.lens_metric import LENS_metric
from metrics.sle_metric import SLE_metric
# from metrics.smart_metric import SMART
# from metrics.matching_functions import BleuMatchingFunction, SARIMatchingFunction
from metrics.my_alignment_metric_v3 import MyAlignmentMetric

def compute_pairwise_metrics(dataset, metrics):
    for metric in metrics:
        cache = {}
        for ind, instance in enumerate(dataset):
            print("instance ", ind)
            org = instance["original"]
            simp1 = instance["simplification1"]
            simp2 = instance["simplification2"]
            refs = instance["references"]

            key = (org, simp1, tuple(refs))
            if key not in cache: 
                cache[key] = metric.compute_metric(org, simp1, refs)
            mval1 = cache[key]

            key = (org, simp2, tuple(refs))
            if key not in cache: 
                cache[key] = metric.compute_metric(org, simp2, refs)
            mval2 = cache[key]

            # mval1 = metric.compute_metric(org, simp1, refs)
            # mval2 = metric.compute_metric(org, simp2, refs)         
            instance.setdefault("metrics", {})
            instance["metrics"].setdefault(metric.name, {})
            instance["metrics"][metric.name]= {
                "simplification1": mval1,
                "simplification2": mval2
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
    parser.add_argument("--lens")
    parser.add_argument("--bert")
    args=parser.parse_args()

    with open(args.dataset) as fp:
        dataset = [json.loads(line.strip()) for line in fp]

    metrics = [
            # SARI(), BLEU(), GLEU(), 
            #    DSARI(), 
            #    SMART(matcher=SARIMatchingFunction()),
            # LENS_metric(args.lens),
            MyAlignmentMetric(args.lens),
            # BERTScore(), 
            # BERTScore(self_flag=True),
            # SLE_metric(True),
            # SLE_metric(False)
    ]
    compute_pairwise_metrics(dataset, metrics)
    # print(dataset[100])

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

'''
TODO
1) Optimize the code to work with this code. Currently taking too much time.
BERTScore(model_name="microsoft/deberta-xlarge-mnli"), 
BERTScore(model_name="microsoft/deberta-xlarge-mnli", self_flag=True)


'''

