import csv
import json
import argparse
import numpy as np

NON_ANSWER_KEYS = ['non_answer_' + str(i) for i in range(1, 4)]
CORRECT_ANSWER_KEYS = ['is_correct_answer_' + str(i) for i in range(1, 4)]


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--human")
    parser.add_argument("--output")
    args=parser.parse_args()

    with open(args.human) as fp:
        csvreader = csv.reader(fp)
        headers = next(csvreader)
        dataset = {}
        for row in csvreader:
            instance = {h:r for h, r in zip(headers, row)}
            key = instance['article_id'], instance['para_id']
            dataset.setdefault(key, {})
            system = instance['Model']
            dataset[key][system] = instance

        with open(args.output, 'w') as fp:
            for _, instance in dataset.items():
                for model, model_info in instance.items():
                    data_json = {}
                    data_json["original"] = instance['Original']['paragraph']
                    data_json["rating_type"] = "single"
                    data_json["references"] = [instance['Elementary']['paragraph']]
                    data_json["simplification"] = model_info['paragraph']
                    data_json["system"] = model_info["Model"]

                    ratings_list = []
                    correct_count = [model_info[key] == 'True' for key in CORRECT_ANSWER_KEYS]
                    assert sum(correct_count) == int(model_info['correct_count'])
                    correct_json = {
                        "name": "Correct answer (out of 3)",
                        "type": "count",
                        "value": correct_count,
                        "agg_value": np.mean(correct_count) * 100.0,
                        "agg_strategy": "percentage"
                    }

                    non_count = [model_info[key] == 'True' for key in NON_ANSWER_KEYS]
                    assert sum(non_count) == int(model_info['non_answer'])
                    non_json = {
                        "name": "Non answer (out of 3)",
                        "type": "count",
                        "value": non_count,
                        "agg_value": -1 * np.mean(non_count) * 100.0,
                        "agg_strategy": "percentage"
                    }

                    data_json['ratings'] = [correct_json, non_json]

                    for key, json_data in data_json.items():
                        print(key)
                        print(json_data)
                    print("---")
                    fp.write(json.dumps(data_json) + "\n")
