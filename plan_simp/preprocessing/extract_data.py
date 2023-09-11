import numpy as np 
import csv
import json
import glob
import argparse
from Levenshtein import ratio

def get_doc_id(id_str):
    return 


def get_valid_complex_sentences(sent_alignments):
   valid_ids = {}
   for _, complex in sent_alignments:
      _, _, para_id, sent_id = complex.split("-")
      para_id = int(para_id)
      sent_id = int(sent_id)

      valid_ids.setdefault(para_id, -1)
      valid_ids[para_id] = max(sent_id, valid_ids[para_id])
   return valid_ids


def is_valid_sentence(ids_str, valid_ids):
     _, _, para_id, sent_id = ids_str.split("-")
     para_id = int(para_id)
     sent_id = int(sent_id)
     max_para = max(valid_ids.keys())
     return para_id in valid_ids and ((para_id == max_para and sent_id <= valid_ids[para_id] + 1) 
                                      or (para_id < max_para))


def map_alignments(sentence_alignments):
    mapping = {}
    for simple, normal in sentence_alignments:
        mapping.setdefault(normal, [])
        mapping[normal].append(simple)
    return mapping


def extract_labels(complex_keys, alignments, complex_content, simple_content):
    labels = []
    for key in complex_keys:
        label = None
        if key not in alignments:
            label = "delete"
        elif key in alignments and len(alignments[key]) > 1:
            label = "ssplit"
        elif key in alignments and len(alignments[key]) == 1:
            similarity = ratio(complex_content[key], 
                               simple_content[alignments[key][0]])
            if similarity <= 0.92:
                label = "rephrase"
            else:
                label = "ignore"
        
        assert label is not None
        labels.append(label)
    return labels
    

def extract_dataset(input_dir):

    field_row = "title,pair_id,complex,simple,labels,c_len,s_len,num_c_sents,del_rate".split(",")
    dataset_rows = [field_row]

    for fpath in glob.glob(input_dir + "/*"):

        data_json = json.load(open(fpath))
        for pair_id, instance in data_json.items():
           pair_id = int(pair_id)

           valid_ids = get_valid_complex_sentences(instance["sentence_alignment"])
           if len(valid_ids) > 0:
            
            complex_json = instance["normal"]["content"]
            complex_keys = sorted(complex_json.keys())
            complex_content = [(key, complex_json[key]) for key in complex_keys if is_valid_sentence(key, valid_ids)]
            complex = " <s> ".join([sent for _, sent in complex_content])

            alignments = map_alignments(instance["sentence_alignment"])
        
            simple = []
            complex_keys = [key for key, _ in complex_content]
            simple_json = instance["simple"]["content"]
            for key in complex_keys:
                if key in alignments:
                    simple_sent_ids = alignments[key]
                    simple_sents = [simple_json[sent_id] for sent_id in simple_sent_ids]
                    simple.extend(simple_sents)
            simple = " <s> ".join(simple)

            labels = extract_labels(complex_keys, alignments, complex_json, simple_json)
            del_rate = np.mean([label == "delete" for label in labels])

            s_len_sent = len(simple.split("<s>"))
            c_len_sent = len(complex.split("<s>"))
            c_len = len(complex.split()) - c_len_sent + 1
            s_len = len(simple.split()) - s_len_sent + 1
            if del_rate <= 0.5 and c_len < 1024 and s_len < 1024:

                title = instance["normal"]['title']
                csv_row = [title, pair_id, complex, simple, labels, c_len, s_len, c_len_sent, del_rate]
                dataset_rows.append(csv_row)

                if pair_id in [71514, 7717, 1657, 111775]:   
                    print()
                    print(complex)
                    print(c_len, s_len, c_len_sent, s_len_sent, del_rate)
                    print(simple)
                    print(labels)
                    print("*******\n")

    return dataset_rows


if __name__ == '__main__':

    # prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input document directory.",
                    type=str)
    parser.add_argument("output", help="Output file.",
                    type=str)
    args = parser.parse_args()

    dataset_rows = extract_dataset(args.input)

    with open(args.output, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(dataset_rows)

    print(len(dataset_rows))
