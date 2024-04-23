import sys

sys.path.append('/Users/mmaddela3/Documents/simplification_evaluation/external_repos/swipe') 

import utils_diff
from model_bic import BIC

import argparse
import numpy as np
from nltk import sent_tokenize
from lens.lens_score import LENS


def get_edits_to_sentences_mapping(edits, edit_types):
    text_wedits = []
    raw_text = []
    for index, edit in enumerate(edits):
        if edit['type'] == edit_types[0] or edit['type'] == edit_types[1]:
            text = edit['text'] if edit['type'] == 'equal' else edit[edit['type']]
            start_label = "[EDIT=" + str(index) + "] " 
            end_label = " [/EDIT=" + str(index) + "]"
            raw_text.append(text.strip())
            text_wedits.append(start_label + text.strip() + end_label)
    
    text_wedits = " ".join(text_wedits).strip()
    text_sents_wedits = sent_tokenize(text_wedits)
    text_sents_wedits = [sent.strip() for sent in text_sents_wedits]

    edit2sents = {}
    for edit_index in range(len(edits)):
        start_label = "[EDIT=" + str(edit_index) + "]" 
        end_label = "[/EDIT=" + str(edit_index) + "]"
        start_index, end_index = -1, -1
        for sent_ind, sent in enumerate(text_sents_wedits):
            if start_label in sent:
                start_index = sent_ind
            if end_label in sent:
                end_index = sent_ind if sent.startswith(end_label) else sent_ind + 1
        edit2sents[edit_index] = (start_index, end_index)

    # print(edit2sents)
    # print(raw_text)
    # print("*" * 30)
    return edit2sents, sent_tokenize(" ".join(raw_text))


def expand_edit_group(edits, edit_group):
    m, M = min(edit_group["opis"]), max(edit_group["opis"])
    opi_range = list(range(m, M+1))
    if opi_range[0] > 0 and edits[opi_range[0]-1]["type"] == "equal":
        opi_range = [opi_range[0]-1] + opi_range
    if opi_range[-1] < len(edits)-1 and edits[opi_range[-1]+1]["type"] == "equal":
        opi_range = opi_range + [opi_range[-1]+1]
    return opi_range


def edit_alignment(bic, textA, textB):
    edit_groups, edits = bic.predict_from_text_pair(textA, textB)
    edit_groups = [expand_edit_group(edits, group) for group in edit_groups]
    edit2sentsA, textA_sents = get_edits_to_sentences_mapping(edits, ['equal', 'delete']) 
    edit2sentsB, textB_sents = get_edits_to_sentences_mapping(edits, ['equal', 'insert'])    
    
    sent_groups = {}
    for ed_group in edit_groups:
        textA_sent_inds = set()
        textB_sent_inds = set()
        for edit_ind in ed_group:
            start, end = edit2sentsA[edit_ind]
            if start >= 0 and end >= 0:
                for i in range(start, end):
                    textA_sent_inds.add(i)
            
            start, end = edit2sentsB[edit_ind]
            if start >= 0 and end >= 0:
                for i in range(start, end):
                    textB_sent_inds.add(i)

        if len(textA_sent_inds) and len(textB_sent_inds):
            textA_sent_inds = tuple(sorted(list(textA_sent_inds)))
            textB_sent_inds = tuple(sorted(list(textB_sent_inds)))
            sent_groups[textA_sent_inds] = textB_sent_inds

    deduped_sent_groups = {}
    for sgroupA_i in sent_groups:
        subset_flag = False
        for sgroupA_j in sent_groups:
            if sgroupA_i != sgroupA_j and len(set(sgroupA_i).difference(set(sgroupA_j))) == 0 \
                and len(set(sent_groups[sgroupA_i]).difference(set(sent_groups[sgroupA_j]))) == 0:
                    subset_flag = True
        if not subset_flag:
            deduped_sent_groups[sgroupA_i] = sent_groups[sgroupA_i]

    
    for k, v in deduped_sent_groups.items():
        print(k, v)
    print("-" * 20)

    return deduped_sent_groups, textA_sents, textB_sents
    

def merge_alignments(alignmentAB, alignmentAC, textA_sents, 
                     textB_sents, textC_sents):
    print("Merge alignments")
    print(len(textA_sents), len(textB_sents), len(textC_sents))
    textA, textB, textC = [], [], []
    for ab_a, ab_b in alignmentAB.items():
        
        new_ab_a = set()
        # new_ab_a.update(ab_a)
        ab_c = set()
        for ac_a, ac_c in alignmentAC.items():
            if len(set(ab_a).intersection(set(ac_a))):
                ab_c.update(ac_c)
                # new_ab_a.update(ac_a)
        print(ab_a, ab_b, ab_c)
        
        

        textA.append(" ".join([textA_sents[ind] for ind in ab_a]).lower().replace(" .", ".").replace(" ,", ","))
        textB.append(" ".join([textB_sents[ind] for ind in ab_b]).lower().replace(" .", ".").replace(" ,", ","))
        textC.append(" ".join([textC_sents[ind] for ind in ab_c if ind < len(textC_sents)]).lower().replace(" .", ".").replace(" ,", ","))
    
    for a, b, c in zip(textA, textB, textC):
        print(a)
        print(b)
        print(c)
        print()

    return textA, textB, textC


class MyAlignmentMetric:

    def __init__(self, lens_path):
        self.name = "MyAligmentMetric-LENS-V3"
        self.bic = BIC("Salesforce/bic_simple_edit_id", device="cpu")
        self.lens_model = LENS(lens_path, rescale=True)
        super().__init__()

    def compute_metric(self, complex, simplified, references):

        complex = complex.replace("\n", " ")
        simplified = simplified.replace("\n", " ")

        ref_scores = []
        for reference in references:
            deduped_sent_groups_CS, textC1_sents, textS_sents = edit_alignment(self.bic, complex, reference)
            deduped_sent_groups_CO, textC2_sents, textO_sents = edit_alignment(self.bic, complex, simplified)
            all_comps, all_cands, all_refs = merge_alignments(deduped_sent_groups_CS, 
                                                                        deduped_sent_groups_CO,
                                                                        textC1_sents, textS_sents, 
                                                                        textO_sents)           
            all_refs = [[ref] for ref in all_refs]
            scores = self.lens_model.score(all_comps, all_cands, all_refs, 
                                        batch_size=1, gpus=0)
            print(scores)
            ref_scores.append(np.mean(scores))
        
        print(ref_scores)
        return max(ref_scores)
    

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument('--LENS', type=str, default='', 
#                         required=True, 
#                         help='Path to the fine-tuned LENS checkpoint.')
    
#     parser.add_argument('--complex')
#     parser.add_argument('--simple')
#     parser.add_argument('--output')
#     args = parser.parse_args()

#     metric = MyAlignmentMetric(args.LENS)
#     complex = open(args.complex).read().strip()
#     output = open(args.output).read().strip()
#     simple = open(args.simple).read().strip()
#     metric.compute_metric(complex, output, [simple])

    