import json
import string
import argparse

import sys
sys.path.append("/Users/mmaddela3/Documents/simplification_evaluation/external_repos/referee/code")
sys.path.append("/Users/mmaddela3/Documents/simplification_evaluation/external_repos/swipe")


from metrics.sari import SARI
from metrics.bleu import BLEU
from metrics.gleu import GLEU
from metrics.D_SARI import DSARI
from metrics.bscore import BERTScore
from metrics.lens_metric import LENS_metric
from metrics.sle_metric import SLE_metric
from metrics.referee import REFEREE
from metrics.agg_metric_graph import AggMeticGraph
from metrics.agg_metric_graph_refless import AggMeticGraphRefless
from metrics.agg_metric_graph_no_complex import AggMeticGraphNoComplex
from metrics.agg_metric_edit_no_complex import AggMeticEditNoComplex
from metrics.agg_metric_edit_refless import AggMeticEditRefless
from metrics.agg_metric_edit import AggMeticEdit
from metrics.smart_eval import SmartScorer

from scipy.stats import pearsonr


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
    parser.add_argument("--lens")
    parser.add_argument("--bert")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    args=parser.parse_args()

    with open(args.dataset) as fp:
        dataset = [json.loads(line.strip()) for line in fp]

        lens_instance = LENS_metric(args.lens)
        referee = REFEREE()
        bert_scorem = BERTScore()
        sle_metric = SLE_metric(True)

        metrics = [
                #    SARI(),
                #    AggMeticGraph(args.bert, SARI()),
                #    BLEU(), 
                #    AggMeticGraphNoComplex(args.bert, BLEU()),
                #    GLEU(),
                #    AggMeticGraphNoComplex(None, GLEU()),
                #    DSARI(),
                #    bert_scorem,
                #    AggMeticGraphNoComplex(None, bert_scorem),
                #    SLE_metric(True),
                #    AggMeticGraphRefless(args.bert, SLE_metric(True)),
                #    SLE_metric(False),
                #    LENS_metric(args.lens),
                #    AggMeticGraph(args.bert, LENS_metric(args.lens)),
                #    referee,
                #    AggMeticGraphRefless(None, referee),

                # AggMeticEditRefless(SLE_metric(True)),
                # AggMeticEditRefless(REFEREE()),
                # AggMeticEditNoComplex(BLEU()),
                # AggMeticEditNoComplex(GLEU()),
                # AggMeticEditNoComplex(BERTScore()),
                # AggMeticEdit(SARI()),
                # AggMeticEdit(LENS_metric(args.lens))

                SmartScorer(matching_fn=BLEU()),
                SmartScorer(matching_fn=BLEU(), final_smart_type='smart2'),
                SmartScorer(matching_fn=BLEU(), final_smart_type='smart1'),
                SmartScorer(matching_fn=GLEU()),
                SmartScorer(matching_fn=GLEU(), final_smart_type='smart2'),
                SmartScorer(matching_fn=GLEU(), final_smart_type='smart1'),
                SmartScorer(bert_scorem),
                SmartScorer(bert_scorem, final_smart_type='smart2'),
                SmartScorer(bert_scorem, final_smart_type='smart1'),
                SmartScorer(referee),
                SmartScorer(referee, final_smart_type='smart2'),
                SmartScorer(referee, final_smart_type='smart1'),
                SmartScorer(sle_metric),
                SmartScorer(sle_metric, final_smart_type='smart2'),
                SmartScorer(sle_metric, final_smart_type='smart1'),
        ]
        
        compute_metrics(dataset, metrics)
        print_pearson(dataset, metrics)
        with open(args.output, 'w') as fp:
            json.dump(dataset, fp)
