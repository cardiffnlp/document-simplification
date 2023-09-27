import random
import json
import argparse


if __name__ == '__main__':

    # prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    dataset = json.load(open(args.input))
    titles = list(dataset.keys())

    random.shuffle(titles)
    print(len(titles))
    
    dataset = {}
    train_cutoff = int(0.75 * len(titles))
    dataset["train"] = titles[:train_cutoff]
    valid_cutoff = train_cutoff + int(0.05 * len(titles))
    dataset["valid"] = titles[train_cutoff: valid_cutoff]
    dataset["test"] = titles[valid_cutoff:]
    with open(args.output, 'w') as f:
        json.dump(dataset, f)

                        


                




