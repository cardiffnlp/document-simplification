import csv
import argparse


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



def print_doc_level_difference(info, my_info):
    if my_info["simple"] != info["simple"]:
        print("Simplifed texts are different:")
        print(info["simple"])
        print(my_info["simple"])
        print()

    if my_info["complex"] != info["complex"]:
        print("Complex texts are different:")
        print(info["complex"])
        print(my_info["complex"])
        print()

            
    if my_info["labels"] != info["labels"]:
        print("Labels are different:")
        print(info["labels"], len(info["labels"]))
        print(my_info["labels"], len(my_info['labels']))
        print()
            
    print(info["c_len"], info["s_len"], info["num_c_sents"], info["del_rate"])
    print(my_info["c_len"], my_info["s_len"], my_info["num_c_sents"], my_info["del_rate"])
    print()
    print("****\n")


def print_sent_level_difference(info, my_info):
    if my_info["simple"] != info["simple"] or info["simp_sent_id"] != my_info["simp_sent_id"]:
        print("Simplifed texts are different:")
        print(info["simp_sent_id"], info["simple"])
        print(my_info["simp_sent_id"], my_info["simple"])
        print()

    if my_info["complex"] != info["complex"] or info["sent_id"] != my_info["sent_id"]:
        print("Complex texts are different:")
        print(info["sent_id"], info["complex"])
        print(my_info["sent_id"], my_info["complex"])
        print()

            
    if my_info["label"] != info["label"]:
        print("Labels are different:")
        print(info["label"], len(info["label"]))
        print(my_info["label"], len(my_info['label']))
        print()
            
    print(info["doc_pos"], info["doc_quint"], info["doc_len"])
    print(my_info["doc_pos"], my_info["doc_quint"], my_info["doc_len"])
    print()
    print("****\n")



if __name__ == '__main__':

    # prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--myfile", type=str)
    parser.add_argument("--reference", type=str)
    parser.add_argument("--doc_level", action="store_true")
    args = parser.parse_args()

    my_dataset = read_file(args.myfile)
    ref_dataset = read_file(args.reference)
    print(len(my_dataset), len(ref_dataset))

    not_equal = 0
    for pair_id, info in ref_dataset.items():

        my_info = my_dataset[pair_id]        
        if args.doc_level:
            my_info["del_rate"] = my_info["del_rate"][:10]
            new_labels = info['labels'].replace('dsplit', 'ssplit')
            info['labels'] = new_labels
            info["del_rate"] = info["del_rate"][:10]
        else:
            new_labels = info['label'].replace('dsplit', 'ssplit')
            info['label'] = new_labels

        if sorted(my_info.items()) != sorted(info.items()):
            print(pair_id)
            if args.doc_level:
                print_doc_level_difference(info, my_info)
            else:
                print_sent_level_difference(info, my_info)
       
            not_equal += 1
    print(not_equal)
