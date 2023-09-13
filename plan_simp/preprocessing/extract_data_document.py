import numpy as np 
import csv
import math
import json
import glob
import argparse
from Levenshtein import ratio

def get_doc_id(id_str):
    return 


def get_valid_complex_sentences(sent_alignments):
   valid_ids = {}
   for _, complex in sent_alignments:

      tokens = complex.split("-")
      para_id = int(tokens[-2])
      sent_id = int(tokens[-1])

      valid_ids.setdefault(para_id, -1)
      valid_ids[para_id] = max(sent_id, valid_ids[para_id])
   return valid_ids


def is_valid_sentence(ids_str, valid_ids):
     tokens = ids_str.split("-")
     para_id = int(tokens[-2])
     sent_id = int(tokens[-1])
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
 

def get_field_names():
    if args.doc_level:
        field_row = "title,pair_id,complex,simple,labels,c_len,s_len,num_c_sents,del_rate".split(",")
    else:
        field_row = "title,pair_id,sent_id,complex,label,simple,simp_sent_id,doc_pos,doc_quint,doc_len".split(",")
    if args.newsela:
        field_row.append("s_level")
    return field_row


def extract_dataset(input_dir):

    dataset_rows = [get_field_names()]

    for fpath in glob.glob(input_dir + "/*"):

        data_json = json.load(open(fpath))

        for pair_id, instance in data_json.items():

           valid_ids = get_valid_complex_sentences(instance["sentence_alignment"])
           if len(valid_ids) > 0:
            
            complex_json = instance["normal"]["content"]
            complex_keys = complex_json.keys()
            # print(valid_ids, complex_keys)
            complex_content = [(key, complex_json[key]) for key in complex_keys 
                               if is_valid_sentence(key, valid_ids)]
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
            if del_rate <= 0.5 and c_len <= 1024:

                title = instance["normal"]['title']
                if args.doc_level:
                    csv_row = [title, pair_id, complex, simple, labels,\
                                c_len, s_len, c_len_sent, del_rate]
                    if args.newsela:
                        csv_row.append(int(pair_id[-1]))
                    
                    dataset_rows.append(csv_row)

                    if pair_id in ["71514", "7717", "1657", "111775"]:   
                        print()
                        print(complex)
                        print(c_len, s_len, c_len_sent, s_len_sent, del_rate)
                        print(simple)
                        print(labels)
                        print("*******\n")
                else:
                    index = 0
                    simp_index = 0
                    for key, label in zip(complex_keys, labels):
                        complex = complex_json[key]
                        simple = []
                        if key in alignments:
                            simple_sent_ids = alignments[key]
                            simple = [simple_json[sent_id] for sent_id in simple_sent_ids]
                        
                        doc_pos = (index + 1) / c_len_sent 
                        width = c_len_sent / 5.0
                        doc_quin = math.ceil((index + 1) / width)
                        csv_row = [title, pair_id, index, complex, label, 
                                   simple, simp_index, doc_pos, doc_quin, c_len_sent]
                        dataset_rows.append(csv_row)
                        
                        if pair_id in ["71514", "7717", "1657", "111775"]:   
                            print()
                            print(complex)
                            print(label, simple)
                            print(index, simp_index, doc_pos, doc_quin, c_len_sent)
                            print("*******\n")

                        index += 1
                        simp_index += len(simple)

    return dataset_rows


if __name__ == '__main__':

    # prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input document directory.",
                    type=str)
    parser.add_argument("output", help="Output file.",
                    type=str)
    parser.add_argument("--newsela", help="Add reading level.",
                    action="store_true")
    parser.add_argument("--doc_level", help="Generate document level data \
                        (otherwise sentence level data is generated).",
                    action="store_true")
    args = parser.parse_args()

    dataset_rows = extract_dataset(args.input)

    with open(args.output, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(dataset_rows)

    print(len(dataset_rows))
