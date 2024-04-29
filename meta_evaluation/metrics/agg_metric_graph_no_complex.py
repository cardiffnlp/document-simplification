import torch
import numpy as np
from nltk import sent_tokenize
from lens.lens_score import LENS
from sklearn.utils.extmath import softmax
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers.data.processors import InputExample, glue_convert_examples_to_features

import logging
logging.disable(logging.WARNING)


def get_score_matrix(ssents, rsents, model, tokenizer, max_length):
    all_sents_tgt = [rsents, ssents]
    all_sents_cand = [ssents, rsents]

    tgt_can_pairs = []
    for tgt_sentences, can_sentences in zip(all_sents_tgt, all_sents_cand):
        for t in tgt_sentences:
            for c in can_sentences:
                tgt_can_pairs.append((t, c))
    tgts, cans = list(zip(*tgt_can_pairs))
    tgts = list(tgts)
    cans = list(cans)
    pairwise_scores = get_similarity_score_from_sent_pair(tgts, cans, 
                                            model, tokenizer, max_length)
    
    nc, nr = len(ssents), len(rsents)
    rc_matrix = np.array(pairwise_scores[:nc * nr]).reshape((nr, nc))
    start_ind = nc * nr
    end_ind = start_ind + nc * nr
    cr_matrix = np.array(pairwise_scores[start_ind: end_ind]).reshape((nc, nr))
    return rc_matrix, cr_matrix


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
    

class AggMeticGraphNoComplex:

    CACHE = {}

    def __init__(self, bert_path, sent_metric):
        self.name = "Aggregation Metric Graph -" + sent_metric.name

        self.tokenizer = BertTokenizer.from_pretrained(bert_path, 
                                        do_lower_case=True)
        self.alignment_model = BertForSequenceClassification.from_pretrained(
                                        bert_path, 
                                        output_hidden_states=True)
        self.alignment_model.eval()
        self.sent_model = sent_metric
        self.cache = AggMeticGraphNoComplex.CACHE
        super().__init__()


    def compute_metric_single(self, complex, simplified, references):

        simplified = simplified.replace("\n", " ")
        ssents = sent_tokenize(simplified)

        ref_scores = []
        for reference in references:
            key = (simplified, reference) 
            if key not in self.cache:

                # print("**"* 20)
                # print(complex)
                # print(simplified)
                # print(reference)
                # print("---" * 20)

                rsents = sent_tokenize(reference)
                # print(len(csents), len(ssents), len(rsents))
                rc_matrix, cr_matrix = get_score_matrix(
                    ssents, rsents, self.alignment_model, self.tokenizer, 128
                )

                adj_list = {}
                consturct_graph(adj_list, rc_matrix, 'r', 's')
                consturct_graph(adj_list, cr_matrix, 's', 'r')

                all_cands, all_refs = [], []
                node_groups = bfs(adj_list)
                for group in node_groups:
                    if len(group) > 1:
                        ss = sorted([int(node[1:]) for node in group if node.startswith('s')])
                        rs = sorted([int(node[1:]) for node in group if node.startswith('r')])
                        ss = " ".join([ssents[i] for i in ss])
                        rs = " ".join([rsents[i] for i in rs])
                        # print(ss)
                        # print(rs)
                        all_cands.append(ss)
                        all_refs.append([rs])
                
                self.cache[key] = [all_cands, all_refs]

            all_cands, all_refs = self.cache[key]
            scores = self.sent_model.compute_metric([], all_cands, all_refs)
            ref_scores.append(np.mean(scores))

        # print(ref_scores)
        return max(ref_scores)
    
    def compute_metric(self, complex, simplified, references) :
        scores = []
        index = 0
        for comp, simp, refs in zip(complex, simplified, references):
            print("Instance ", index, len(complex))
            scores.append(self.compute_metric_single(comp, simp, refs))
            index += 1
        return scores
    