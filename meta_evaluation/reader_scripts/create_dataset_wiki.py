"""Integrates human evaluation data in the paper https://arxiv.org/pdf/2110.05071.pdf."""

import csv
import glob
import json
import argparse
import numpy as np
from fuzzywuzzy import fuzz


def clean_string(input):
    input = input.replace(" .", ".")
    input = input.replace(" ,", ",")
    input = input.replace(" )", ")")
    input = input.replace("( ", "(")
    input = input.replace(" ]", "]")
    input = input.replace("[ ", "[")
    input = input.replace("`` ", '"')
    input = input.replace(" ''", '"')
    input = input.replace("( )", '()')
    input = input.replace(" 's", "'s")
    return input.strip()


def read_references(dataset_prefix, original):
    original = set(original)
    references = {}
    with open(dataset_prefix + ".src") as fps:
        with open(dataset_prefix + ".tgt") as fpt:
            for src, tgt in zip(fps, fpt):
                src = clean_string(src)
                tgt = clean_string(tgt)

                final_src = None
                if src in original:
                    final_src = src
                else:
                    for org in original:
                        if fuzz.ratio(org, src) > 90.0:
                            final_src = org
                            break
                if final_src is not None:
                    references.setdefault(final_src, [])
                    references[final_src].append(tgt)

    print(len(references))
    return references


def read_ratings(human_eval_folder):
    all_ratings = {}
    for fpath in glob.glob(human_eval_folder + "/*"):
        with open(fpath, 'r', encoding='ISO-8859-1') as fp:
            csv_reader = csv.reader(fp)  
            _ = next(csv_reader) 

            for index, row in enumerate(csv_reader):
                if index % 8 == 0:
                    raw_original = clean_string(row[1])
                    original = raw_original
                    if raw_original not in all_ratings:
                        for org in all_ratings:
                            if fuzz.ratio(org, raw_original) > 90.0:
                                original = org
                                break
                    all_ratings.setdefault(original, {})
                    
                elif 1 <= index % 8 <= 6:
                    simplification = clean_string(row[1])

                    all_ratings[original].setdefault(simplification, {})
                    all_ratings[original][simplification].setdefault('simplicity-word', [])
                    all_ratings[original][simplification].setdefault('simplicity-sent', [])
                    all_ratings[original][simplification].setdefault('meaning', [])
                    all_ratings[original][simplification].setdefault('grammar', [])
                    all_ratings[original][simplification].setdefault('overall', [])

                    
                    all_ratings[original][simplification]['simplicity-word'].append(int(row[2]))
                    all_ratings[original][simplification]['simplicity-sent'].append(int(row[3]))
                    all_ratings[original][simplification]['meaning'].append(int(row[4]))
                    all_ratings[original][simplification]['grammar'].append(int(row[5]))
                    all_ratings[original][simplification]['overall'].append(int(row[6]))
    return all_ratings


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--prefix")
    parser.add_argument("--human")
    parser.add_argument("--output")
    args=parser.parse_args()

    all_ratings = read_ratings(args.human)
    references = read_references(args.prefix, list(all_ratings.keys()))

    with open(args.output, 'w') as fp:
        for original, info in all_ratings.items():
            for simplification, ratings_info in info.items():
                data_json = {}
                data_json["original"] = original
                data_json["rating_type"] = "single"
                data_json["references"] = references[original]
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
