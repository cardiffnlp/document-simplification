import csv

def read_file(file_str):
    final_rows = {}
    with open(file_str, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            new_row = {}
            for field, value in zip(header, row):
                new_row[field] = value
            final_rows[new_row["pair_id"]] = new_row
    return final_rows