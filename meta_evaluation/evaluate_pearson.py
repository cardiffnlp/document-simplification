import json
import string
import argparse

from scipy.stats import pearsonr

import sys
sys.path.append("/home/ubuntu/simplification/external/referee/code")
sys.path.append("/home/ubuntu/simplification/external/sle-main")

from metrics.sari import SARI
from metrics.bleu import BLEU
from metrics.D_SARI import DSARI
from metrics.referee import REFEREE
from metrics.bscore import BERTScore
from metrics.sle_metric import SLE_metric
from metrics.lens_metric import LENS_metric
from metrics.agg_metric_graph_v2 import AggMeticGraph

from transformers import LlamaForCausalLM, AutoTokenizer
from metrics.llama_metric import LLAMA_metric


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
    parser.add_argument("--bert")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    args=parser.parse_args()

    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = LlamaForCausalLM.from_pretrained(model_id)

    with open(args.dataset) as fp:
        dataset = [json.loads(line.strip()) for line in fp]
        metrics = [
                   SARI(),
                   AggMeticGraph(args.bert, SARI()),
                   BLEU(), 
                   DSARI(),
                   BERTScore(),
                   AggMeticGraph(None, BERTScore()),
                   SLE_metric(True),
                   AggMeticGraph(None, SLE_metric(True), refless=True),
                   LENS_metric(),
                   AggMeticGraph(None, LENS_metric()),
                   REFEREE(),
                   AggMeticGraph(None, REFEREE(), refless=True),

                  # LLAMA_metric(model, tokenizer, "prompts/qa/meaning_preservation_one_shot.txt"),
                  # LLAMA_metric(model, tokenizer, "prompts/meaning_preservation.txt")
        ]
        
        compute_metrics(dataset, metrics)
        print_pearson(dataset, metrics)
        with open(args.output, 'w') as fp:
            json.dump(dataset, fp)
