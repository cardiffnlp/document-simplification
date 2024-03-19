"""Integrates human evaluation data in the paper https://aclanthology.org/2023.findings-emnlp.322."""

import csv
import glob
import json
import argparse
from bs4 import BeautifulSoup


def read_html(html_file):
    with open(html_file) as fp:
        html_doc = fp.read()
        soup = BeautifulSoup(html_doc, 'html.parser')
    
        indices = [int(result.text.strip().split()[-1]) 
                   for result in soup.find_all("h1")]
        
        index = 0
        original, bart_xsum = [], []
        for result in soup.find_all("p"):
            if index % 2 == 0:
                original.append(result.text.strip())
            else:
                bart_xsum.append(result.text.strip())
            index+=1
        assert index == 2*len(original)
        assert len(original) == len(bart_xsum)


        index = 0
        model_versions = [[], [], [], []]
        for result in soup.find_all("li"):
            text = result.text.strip()
            model_versions[(index % 4)].append(text)
            index += 1
        assert index == 4*len(original)
        assert len(original) == len(model_versions[0]) \
            == len(model_versions[1]) == len(model_versions[2]) \
            == len(model_versions[3])

        
        html_content = {}
        for ind, index in enumerate(indices):

            simps = {}
            for version in model_versions:
                simp_tokens = version[ind].split(":")
                model = simp_tokens[0]
                text = ":".join(simp_tokens[1:])
                simps[model] = text.strip()

            obj = {
                "original": original[ind],
                "bart-xsum": bart_xsum[ind],
                "other_models": simps
            }
            html_content[index] = obj

    return html_content


def read_references(dataset_prefix):
    references = {}
    with open(dataset_prefix + ".source") as fps:
        with open(dataset_prefix + ".target") as fpt:
            for src, tgt in zip(fps, fpt):
                src = src.strip()
                tgt = tgt.strip()
                references.setdefault(src, [])
                references[src].append(tgt)
    return references


def read_system_outputs(output_folder):
    system_outputs = {}
    for fpath in glob.glob(output_folder + "/*"):
        system = fpath.split("/")[-1]
        with open(fpath) as fp:
            lines = [line.strip() for line in fp]
            system_outputs[system] = lines    
    return system_outputs


def read_ratings(human_eval_folder):
    all_ratings = {}
    for fpath in glob.glob(human_eval_folder + "/*"):
        with open(fpath) as fp:
            csv_reader = csv.reader(fp)
            _ = next(csv_reader)
            header = next(csv_reader)
            for row in csv_reader:
                index = int(row[0])
                for system, rating in zip(header[1:], row[1:]):
                    all_ratings.setdefault(index, {})
                    all_ratings[index].setdefault(system, [])
                    all_ratings[index][system].append(int(rating))
    return all_ratings



if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--html")
    parser.add_argument("--prefix")
    parser.add_argument("--human")
    parser.add_argument("--outputs")
    parser.add_argument("--verbose", action = "store_true")
    args=parser.parse_args()

    html_content = read_html(args.html)
    references = read_references(args.prefix)
    system_outputs = read_system_outputs(args.outputs)
    human_ratings = read_ratings(args.human)

    if args.verbose:
        for key, content in html_content.items():
            print(key)
            print(content["original"])
            print(content["bart-xsum"])
            assigned = 0
            for model, output in content["other_models"].items():
                for system, outputs in system_outputs.items():
                    if output == outputs[key]:
                        print(model + " is " + system)
                        assigned += 1
                print(output)
            assert assigned == 3
            assert len(human_ratings[key]) == 4
            print(human_ratings[key])
            print("******************")
        
        print("Verified system outputs and ratings")


    for key, content in html_content.items():

        for model, output in content["other_models"].items():

            data_json = {}
            data_json["original"] = content["original"]
            data_json["rating_type"] = "pairwise"
            data_json["references"] = references[content["original"]]
            data_json["simplification1"] = content["bart-xsum"]
            data_json["simplification2"] = output

            final_system = ""
            for system, sys_outputs in system_outputs.items():
                if output == sys_outputs[key]:
                    final_system = system
            if len(final_system) == 0:
                final_system = "UL"
            
            data_json["system1"] = "bart-xsum"
            data_json["system2"] = final_system
            rating_values = human_ratings[key][final_system]
            ratings = {
                "name": "readability",
                "type": "binary",
                "value": rating_values,
                "agg_value": 1 if sum(rating_values) > 1 else 0,
                "agg_strategy": "majority"
            }
            data_json["ratings"] = [ratings]

            print(json.dumps(data_json)) 
            







