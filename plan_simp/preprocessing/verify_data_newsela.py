import ast
import json
import argparse

from utils import read_file


if __name__ == '__main__':

    # prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--myfile", type=str)
    parser.add_argument("--reference", type=str)
    parser.add_argument("--doc_level", action="store_true")
    args = parser.parse_args()

    my_dataset = read_file(args.myfile)
    ref_dataset = json.load(open(args.reference))
    print(len(my_dataset), len(ref_dataset))

    for pair_id, my_info in my_dataset.items():
        article_id = pair_id[:-4]
        c_id, s_id = pair_id[-3], pair_id[-1]

        info = ref_dataset[article_id]
        alignments = info["sentence_alignment"]
        complex_sents = set(info[c_id].values())
        simple_sents = set(info[s_id].values())
        aligned_complex = set([info[c_id][ac_id] for ac_id, sc_id in alignments \
                               if ("en-" + c_id in ac_id) and ("en-" + s_id in sc_id)])

        if args.doc_level:
            complex = my_info["complex"].split(" <s> ")
            simple = my_info["simple"].split(" <s> ")
            assert len(set(complex).difference(complex_sents)) == 0
            assert len(set(simple).difference(simple_sents)) == 0
            assert len(aligned_complex.difference(set(complex))) == 0
            assert my_info["s_level"] == s_id
        else:
            complex = my_info["complex"]
            simple = my_info["simple"]
            simple = ast.literal_eval(simple)
            assert complex in complex_sents
            # print(isinstance(simple, list))
            # print(article_id)
            # print(simple)
            # print(simple_sents)
            assert len(set(simple).difference(simple_sents)) == 0
            assert my_info["s_level"] == s_id

