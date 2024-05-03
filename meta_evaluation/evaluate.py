import json
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
                if mvals["simplification2"] > mvals["simplification1"]:
                    pairwise_mval = 1
                human = rating["agg_value"]
                
                if pairwise_mval == human:
                    agg_ratings_dimension[name][metric]["concordant"] += 1
                else:
                    agg_ratings_dimension[name][metric]["discordant"] += 1
    
    return agg_ratings_dimension


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--lens")
    parser.add_argument("--bert")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    args=parser.parse_args()

    with open(args.dataset) as fp:
        dataset = [json.loads(line.strip()) for line in fp]

        # lens_instance = LENS_metric(args.lens)
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
                   LENS_metric(args.lens),
                   AggMeticGraph(args.bert, LENS_metric(args.lens)),
                   AggMeticGraph(None, LENS_metric(args.lens), weighted=True),
                #    referee,
                #    AggMeticGraphRefless(None, referee),
                # AggMeticEditRefless(SLE_metric(True)),
                # AggMeticEditRefless(REFEREE()),
                # AggMeticEditNoComplex(BLEU()),
                # AggMeticEditNoComplex(GLEU()),
                # AggMeticEditNoComplex(BERTScore()),
                # AggMeticEdit(SARI()),
                # AggMeticEdit(LENS_metric(args.lens))
                    # SmartScorer(matching_fn=BLEU()),
                    # SmartScorer(matching_fn=BLEU(), final_smart_type='smart2'),
                    # SmartScorer(matching_fn=BLEU(), final_smart_type='smart1'),
                    # SmartScorer(GLEU()),
                    # SmartScorer(GLEU(), final_smart_type='smart2'),
                    # SmartScorer(GLEU(), final_smart_type='smart1'),
                    # SmartScorer(bert_scorem),
                    # SmartScorer(bert_scorem, final_smart_type='smart2'),
                    # SmartScorer(bert_scorem, final_smart_type='smart1'),
                    # SmartScorer(referee),
                    # SmartScorer(referee, final_smart_type='smart2'),
                    # SmartScorer(referee, final_smart_type='smart1'),
                    # SmartScorer(sle_metric),
                    # SmartScorer(sle_metric, final_smart_type='smart2'),
                    # SmartScorer(sle_metric, final_smart_type='smart1'),

        ]
        
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

        with open(args.output, 'w') as fp:
            json.dump(dataset, fp)


