import glob
import json
import random
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    ## Preparing OneStopEnglish Corpus
    dataset = {}
    file_paths = glob.glob(args.root + '/*/*', recursive=True)
    for file_path in file_paths:
        filename = file_path.split("/")[-1]
        print(filename.split("-"))
        filename_tokens = filename.split("-")
        filename = "-".join(filename_tokens[:-1])
        level = filename_tokens[-1][:3]

        file_text = open(file_path).read().strip()
        dataset.setdefault(filename, {})
        dataset[filename][level] = file_text

    new_dataset = []
    for fname, contents in dataset.items():
        instance = {
            "r_content": contents['adv'],
            "s_content": [contents['int'], contents['ele']]
        }
        new_dataset.append(instance)

    random.shuffle(new_dataset)
    
    with open(args.output + "/val.json", "w") as outfile:
        json.dump(new_dataset[:19], outfile)
    
    with open(args.output + "/test.json", "w") as outfile:
        json.dump(new_dataset[19:], outfile)

    # ## Preparing Cochrane Corpus
    # new_dataset = []
    # fs = open(args.root + ".source")
    # ft = open(args.root + ".target")
    # for complex, simple in zip(fs, ft):
    #     complex = complex.strip()
    #     simple = simple.strip()
    #     instance = {
    #         "r_content": complex,
    #         "s_content": [simple]
    #     }
    #     new_dataset.append(instance)
    #     print(complex)
    #     print(simple)

    
    # with open(args.output, "w") as outfile:
        # json.dump(new_dataset, outfile)
