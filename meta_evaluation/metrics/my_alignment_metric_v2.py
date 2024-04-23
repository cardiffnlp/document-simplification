import torch
import argparse
import numpy as np
from nltk import sent_tokenize
from lens.lens_score import LENS
from sklearn.utils.extmath import softmax
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers.data.processors import InputExample, glue_convert_examples_to_features


import bert_score

def get_score_matrix(tgt_sentences, can_sentences, model, tokenizer, max_length):
    tgt_can_pairs = []
    for t in tgt_sentences:
      for c in can_sentences:
        tgt_can_pairs.append((t, c))
    tgts, cans = list(zip(*tgt_can_pairs))
    tgts = list(tgts)
    cans = list(cans)
    pairwise_scores = get_similarity_score_from_sent_pair(tgts, cans, 
                                            model, tokenizer, max_length)
    # (_, _, pairwise_scores), _ = bert_score.score(cans, tgts, lang="en", 
                                        # return_hash=True, verbose=True, idf=False,
                                        # model_type="roberta-large")
    
    score_matrix = np.array(pairwise_scores).reshape(
        (len(tgt_sentences), len(can_sentences)))
    return score_matrix


def get_similarity_score_from_sent_pair(sentA_list, sentB_list, model, 
                                        tokenizer, max_length = 128):

    model.eval()
    fake_example = []

    for i in range(len(sentA_list)):
        fake_example.append(InputExample(guid=i, text_a=sentA_list[i], text_b=sentB_list[i], label='good'))

    fake_example_features = glue_convert_examples_to_features(fake_example, tokenizer, max_length, label_list = ["good", 'bad'], output_mode = 'classification')

    all_input_ids = torch.tensor([f.input_ids for f in fake_example_features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in fake_example_features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in fake_example_features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in fake_example_features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)
    
    output_tensor = []
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=8)
    for batch in eval_dataloader:
        # my_device = torch.device('cuda:0')
        # batch = tuple(t.to(my_device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(input_ids=inputs["input_ids"], \
                            attention_mask=inputs["attention_mask"], \
                            token_type_ids=inputs["token_type_ids"], \
                            labels=None, \
                            )
            output_tensor.append(outputs['logits'].cpu().data)

    output_tensor = torch.cat(output_tensor)
    probabilities = softmax(output_tensor)
    probabilities = [i[0] for i in probabilities]
    return probabilities

def consturct_graph(adj_list, matrix, label_row, label_col):
    m,n = matrix.shape
    print(matrix)
    for i in range(m):
        x = label_row + str(i)
        for j in range(n):
            y = label_col + str(j)
            if matrix[i][j] > 0.5:
                adj_list.setdefault(x, [])
                adj_list[x].append(y)
            
def bfs(adj_list):
    node_groups = []
    queue = []
    visited = set()

    for key in adj_list:
        if key not in visited:
            queue.append(key)
            node_group = set()
            while len(queue) > 0:
                node = queue.pop()
                visited.add(node)
                node_group.add(node)
                for child in adj_list.get(node, []):
                    if child not in visited:
                        queue.append(child)
            print(key, node_group)
            node_groups.append(node_group)
    return node_groups
    

class MyAlignmentMetric:

    def __init__(self, bert_path, lens_path):
        self.name = "MyAligmentMetricV2-LENS"
        self.tokenizer = BertTokenizer.from_pretrained(bert_path, 
                                        do_lower_case=True)
        self.alignment_model = BertForSequenceClassification.from_pretrained(
                                        bert_path, 
                                        output_hidden_states=True)
        self.alignment_model.eval()
        self.lens_model = LENS(lens_path, rescale=True)
        super().__init__()

    def compute_metric(self, complex, simplified, references):

        complex = complex.replace("\n", " ")
        simplified = simplified.replace("\n", " ")
        csents = sent_tokenize(complex)
        ssents = sent_tokenize(simplified)

        ref_scores = []
        for reference in references:

            print("**"* 20)
            print(complex)
            print(simplified)
            print(reference)
            print("---" * 20)

            rsents = sent_tokenize(reference)
            print(len(csents), len(ssents), len(rsents))

            adj_list = {}
            rc_matrix = get_score_matrix(rsents, csents, self.alignment_model, 
                                     self.tokenizer, 128)
            consturct_graph(adj_list, rc_matrix, 'r', 'c')
            cr_matrix = get_score_matrix(csents, rsents, self.alignment_model, 
                                     self.tokenizer, 128)
            consturct_graph(adj_list, cr_matrix, 'c', 'r')
            sc_matrix = get_score_matrix(ssents, csents, self.alignment_model, 
                                     self.tokenizer, 128)
            consturct_graph(adj_list, sc_matrix, 's', 'c')
            cs_matrix = get_score_matrix(csents, ssents, self.alignment_model, 
                                     self.tokenizer, 128)
            consturct_graph(adj_list, cs_matrix, 'c', 's')
            # rs_matrix = get_score_matrix(rsents, ssents, self.alignment_model, 
            #                          self.tokenizer, 128)
            # consturct_graph(adj_list, rs_matrix, 'r', 's')
            # sr_matrix = get_score_matrix(ssents, rsents, self.alignment_model, 
            #                          self.tokenizer, 128)
            # consturct_graph(adj_list, sr_matrix, 's', 'r')
            all_comps, all_cands, all_refs = [], [], []
            print(adj_list)
            node_groups = bfs(adj_list)
            for group in node_groups:
                if len(group) > 2:
                    cs = sorted([int(node[1:]) for node in group if node.startswith('c')])
                    ss = sorted([int(node[1:]) for node in group if node.startswith('s')])
                    rs = sorted([int(node[1:]) for node in group if node.startswith('r')])
                    print(cs, ss, rs)
                    cs = " ".join([csents[i] for i in cs]).lower()
                    ss = " ".join([ssents[i] for i in ss]).lower()
                    rs = " ".join([rsents[i] for i in rs]).lower()
                    print(cs)
                    print(ss)
                    print(rs)
                    if True: # len(cs) > 0 and len(ss) > 0 and len(rs) > 0:
                        all_comps.append(cs)
                        all_cands.append(ss)
                        all_refs.append([rs])

            scores = self.lens_model.score(all_comps, all_cands, all_refs, 
                                        batch_size=1, gpus=0)
            print(scores, np.mean(scores))
            ref_scores.append(np.mean(scores))

        print(ref_scores)
        return max(ref_scores)
    

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument('--BERT', type=str, default='', 
#                         required=True, 
#                         help='Path to the fine-tuned BERT folder.')
#     parser.add_argument('--LENS', type=str, default='', 
#                         required=True, 
#                         help='Path to the fine-tuned LENS checkpoint.')
    
#     parser.add_argument('--complex')
#     parser.add_argument('--simple')
#     parser.add_argument('--output')
#     args = parser.parse_args()

#     metric = MyAlignmentMetricV2(args.BERT, args.LENS)
#     complex = open(args.complex).read().strip()
#     output = open(args.output).read().strip()
#     simple = open(args.simple).read().strip()
#     metric.compute_metric(complex, output, [simple])
    