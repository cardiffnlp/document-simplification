"""Integrates human evaluation data in the paper https://arxiv.org/pdf/2110.05071.pdf."""

import re
import glob
import json
import argparse
import numpy as np
from fuzzywuzzy import fuzz


def clean_string(input):
    input = " ".join([sent.strip() for sent in input.split("\n")])
    return input.strip()


def read_references(dataset_prefix):
    references = {}
    with open(dataset_prefix + ".source") as fps:
        with open(dataset_prefix + ".target") as fpt:
            for src, tgt in zip(fps, fpt):
                src = clean_string(src)
                tgt = clean_string(tgt)
                references.setdefault(src, [])
                references[src].append(tgt)
    print(len(references))
    return references


def read_ratings(human_eval_folder):

    all_ratings = {}
    for fpath in glob.glob(human_eval_folder + "/*"):
        with open(fpath) as fp:
            json_data = json.load(open(fpath))
            for instance in json_data:
                text = instance['text']
                original, simplification = text.split("Simplified text:\n")
                simplification = clean_string(simplification)
                original = clean_string(original.replace("Unsimplified text:\n", ""))

                all_ratings.setdefault(original, {})
                all_ratings[original].setdefault(simplification, {})
                
                for label_str in instance['label']:
                    dimension, label = label_str.split()
                    label = int(label)
                    all_ratings[original][simplification].setdefault(dimension, [])
                    all_ratings[original][simplification][dimension].append(label)
    
    print(len(all_ratings))
    return all_ratings

def get_correct_reference(original, references):
    if original in references:
        return references[original]
    else:
        for org in references:
            if fuzz.ratio(org, original) > 98.0:
                return references[org]
    return []



if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--prefix")
    parser.add_argument("--human")
    parser.add_argument("--output")
    args=parser.parse_args()

    all_ratings = read_ratings(args.human)
    references = read_references(args.prefix)

    with open(args.output, 'w') as fp:
        for original, info in all_ratings.items():
            for simplification, ratings_info in info.items():
                data_json = {}
                data_json["original"] = original
                data_json["rating_type"] = "single"

                refs = get_correct_reference(original , references)
                data_json["references"] = refs
                assert len(refs) > 0
                
                data_json["simplification"] = simplification
                ratings_list = []
                for dimension, raw_ratings in ratings_info.items():
                    ratings = {
                        "name": dimension,
                        "type": "likert",
                        "value": raw_ratings,
                        "agg_value": np.mean(raw_ratings),
                        "agg_strategy": "average"
                    }
                    ratings_list.append(ratings)
                data_json['ratings'] = ratings_list

            fp.write(json.dumps(data_json) + "\n")
