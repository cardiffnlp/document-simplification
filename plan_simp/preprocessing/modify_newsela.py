import random
import csv
import json
import argparse

if __name__ == '__main__':

    # prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    
    with open(args.input, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        header.append("c_level")
        csv_rows = [header]
        for row in reader:
            c_level = row[0][-1]
            row.append(c_level)
            csv_rows.append(row) 
       
    with open(args.output, 'w') as fo:
        writer = csv.writer(fo)
        writer.writerows(csv_rows)

                        


                




