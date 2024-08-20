import json
import argparse

from scipy.stats import pearsonr
            
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
                if mvals["simplification2"] > mvals["simplification1"]:
                    pairwise_mval = 1
                human = rating["agg_value"]
                
                if pairwise_mval == human:
                    agg_ratings_dimension[name][metric]["concordant"] += 1
                else:
                    agg_ratings_dimension[name][metric]["discordant"] += 1
    
    return agg_ratings_dimension


def print_pearson(dataset):
    dimensions = [rating["name"] for rating in dataset[0]["ratings"]]
    metrics = dataset[0]["metrics"].keys()
    for dimension in dimensions:
        print(dimension)
        for metric in metrics:
            mvals, human_vals = [], []
            for instance in dataset:
                mvals.append(instance["metrics"][metric])
                for ratings in instance['ratings']:
                    if ratings['name'] == dimension:
                        ratings_val = ratings["agg_value"]
                        break
                human_vals.append(ratings_val)
            corr, _ = pearsonr(mvals, human_vals)
            print(metric, corr)
            

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--type")
    args=parser.parse_args()

    from utils import pairwise_kendall, pointwise_pearson
    metrics = ["Aggregation Metric Graph Sentence -LENS", 
               "Aggregation Metric Graph Sentence -SARI",
               "Aggregation Metric Graph Sentence -BERTScore-ref-roberta-large",
               "Aggregation Metric Graph Sentence -REFEREE"]
    # metrics = ["Aggregation Metric Graph -LENS", 
    #            "Aggregation Metric Graph -SARI",
    #            "Aggregation Metric Graph -BERTScore-ref-roberta-large",
    #            "Aggregation Metric Graph -REFEREE"]
    with open(args.dataset) as fp:
        dataset = json.loads(fp.read().strip())
        for metric in metrics:
            print(metric, pointwise_pearson(dataset, metric, dimensions=['meaning', 'grammar', 'simplicity-overall']))
            # print(metric, pairwise_kendall(dataset, metric))
    

    # with open(args.dataset) as fp:
    #         dataset = json.loads(fp.read().strip())

    #         if args.type == "pairwise":
    #             correlation_values = pairwise_kendall(dataset)
    #             for dimen in correlation_values:
    #                 print(dimen)
    #                 for metric, counts in correlation_values[dimen].items():
    #                     print(metric)
    #                     concordant = counts["concordant"]
    #                     discordant = counts["discordant"]
    #                     if (concordant + discordant) > 0:
    #                         tau = (concordant - discordant) / (concordant + discordant)
    #                         print(tau, concordant * 100.0 / (concordant + discordant), concordant, discordant)
        