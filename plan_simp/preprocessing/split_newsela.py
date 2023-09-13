import random
import csv
import json
import argparse

from utils import read_file


if __name__ == '__main__':

    # prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--split_file", type=str)
    parser.add_argument("--split_dataset", action="store_true")
    args = parser.parse_args()

    dataset = read_file(args.input)

    if args.split_dataset:
        titles = list(set([row["title"][:-2] for row in dataset.values()]))
        random.shuffle(titles)
        print(len(titles))
        dataset = {}
        train_cutoff = int(0.925 * len(titles))
        dataset["train"] = titles[:train_cutoff]
        valid_cutoff = train_cutoff + int(0.025 * len(titles))
        dataset["valid"] = titles[train_cutoff: valid_cutoff]
        dataset["test"] = titles[valid_cutoff:]
        with open(args.output, 'w') as f:
            json.dump(dataset, f)
    
    else:
        split_files = json.load(open(args.split_file))
        for split in ["train", "valid", "test"]:
            print(len(split_files[split]))
            with open(args.input) as f:
                reader = csv.reader(f)
                header = next(reader)
                csv_rows = [header]
                for row in reader:
                    title = row[0][:-2]
                    if title in split_files[split]:
                        csv_rows.append(row)
                with open(args.output + split + ".csv", 'w') as fo:
                    writer = csv.writer(fo)
                    writer.writerows(csv_rows)

                        


                




