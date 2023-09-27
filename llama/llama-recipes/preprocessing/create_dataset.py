import random
import json
import argparse


def get_paragraphs(instance):
    dict_text = {}
    for key, sentence in instance.items():
        tokens = key.split("-")
        para =  "-".join(tokens[:-1])
        sent_id = int(tokens[-1])
        dict_text.setdefault(para, {})
        dict_text[para][sent_id] = sentence

    new_dict_text = {}
    for para, sent_dict in dict_text.items():
        sentences = []
        for i in range(len(sent_dict)):
            sentences.append(sent_dict[i])
        new_dict_text[para] = " ".join(sentences)
    return new_dict_text



def get_non_adjacent_alignments(al_dict, paraid):
    paraids = [paraid]
    for i in range(3):
        next_level = []
        for para in paraids:
            next_level.extend(al_dict.get(para, []))
        
        paraids = next_level
    return paraids


if __name__ == '__main__':

    # prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--split_file", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    split = args.split
    dataset = json.load(open(args.input))
    splits = json.load(open(args.split_file))

    index = 0
    output_dataset = []
    for fname in splits[split]:
        instance = dataset[fname]
        complex = get_paragraphs(instance['0'])
        simple = get_paragraphs(instance['3'])
        
        adj_alignments = {}
        for cid, sid in instance['paragraph_alignment']:
            adj_alignments.setdefault(cid, [])
            adj_alignments[cid].append(sid)

        for para in complex.keys():
            simp_alignments = get_non_adjacent_alignments(adj_alignments, para)
            if len(simp_alignments) > 0:
                
                complex_text = complex[para]
                simple_text = " ".join([simple[spara] for spara in simp_alignments])
                ratio = len(simple_text) * 1.0 / len(complex_text)
                instance_json = {
                        "id": index,
                        "r_content": complex_text,
                        "s_content": simple_text
                    }
                
                if 0.2 <= ratio <= 2.0:
                    index += 1
                    output_dataset.append(instance_json)
                    # print(para)
                    # print(simp_alignments)
                    # print(instance_json)
                    # print()
    
    with open(args.output, 'w') as fp:
        json.dump(output_dataset, fp)
                


                        


                




