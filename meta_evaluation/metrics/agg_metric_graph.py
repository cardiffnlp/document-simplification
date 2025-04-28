import torch
import numpy as np
from nltk import sent_tokenize
from sklearn.utils.extmath import softmax
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers.data.processors import InputExample, glue_convert_examples_to_features

import logging
logging.disable(logging.WARNING)


def get_score_matrix(csents, rsents, ssents, model, tokenizer, max_length):
    all_sents_tgt = [rsents, csents, ssents, csents]
    all_sents_cand = [csents, rsents, csents, ssents]

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
    
    nc, ns, nr = len(csents), len(ssents), len(rsents)
    rc_matrix = np.array(pairwise_scores[:nc * nr]).reshape((nr, nc))
    start_ind = nc * nr
    end_ind = start_ind + nc * nr
    cr_matrix = np.array(pairwise_scores[start_ind: end_ind]).reshape((nc, nr))
    start_ind = end_ind
    end_ind = start_ind + ns * nc
    sc_matrix = np.array(pairwise_scores[start_ind: end_ind]).reshape((ns, nc))
    cs_matrix = np.array(pairwise_scores[-1 * nc * ns:]).reshape((nc, ns))
    return rc_matrix, cr_matrix, sc_matrix, cs_matrix


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
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=64)
    for batch in eval_dataloader:
        my_device = torch.device('cuda:0')
        batch = tuple(t.to(my_device) for t in batch)
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


def consturct_graph(adj_list, matrix, label_row, label_col, threshold):
    m,n = matrix.shape
    for i in range(m):
        x = label_row + str(i)
        for j in range(n):
            y = label_col + str(j)
            adj_list.setdefault(x, [])
            adj_list.setdefault(y, [])
            if matrix[i][j] > threshold:
                adj_list[x].append(y)
                adj_list[y].append(x)


def merge_node_groups(node_groups):
    single_nodes = []
    new_node_groups = []
    for group in node_groups:
        if len(group) > 1:
            new_node_groups.append(group)
        else:
            single_nodes.append(group)

    for start in ['c', 's', 'r']:
        single_nodes_of_type = []
        for node in single_nodes:
            node = list(node)[0]
            if node.startswith(start):
                num = int(node[1:])
                single_nodes_of_type.append((num, num + 1))
        single_nodes_of_type = sorted(single_nodes_of_type)

        while len(single_nodes_of_type) > 1:
            if single_nodes_of_type[0][1] == single_nodes_of_type[1][0]:
                node = single_nodes_of_type.pop(0)
                single_nodes_of_type[0] = (node[0], single_nodes_of_type[0][1])
            else:
                node = single_nodes_of_type.pop(0)
                new_node_groups.append(set([start + str(i) for i in range(node[0], node[1])]))

        if  len(single_nodes_of_type) == 1:
                node = single_nodes_of_type.pop(0)
                new_node_groups.append(set([start + str(i) for i in range(node[0], node[1])]))

    return new_node_groups
          
def bfs(adj_list):
    node_groups = []
    queue = []
    visited = set()

    for key in adj_list:
        if key not in visited and (key.startswith('s') or key.startswith('c')):
            queue.append(key)
            node_group = set()
            while len(queue) > 0:
                node = queue.pop()
                visited.add(node)
                node_group.add(node)
                for child in adj_list.get(node, []):
                    if child not in visited:
                        queue.append(child)
            node_groups.append(node_group)
    return node_groups
    

class AggMetricGraph:

    CACHE = {}

    def __init__(self, bert_path, sent_metric, refless=False, threshold=0.5):
        self.name = "Aggregation Metric Graph -" + sent_metric.name
        if bert_path is not None:
            self.tokenizer = BertTokenizer.from_pretrained(bert_path, 
                                            do_lower_case=True)
            self.alignment_model = BertForSequenceClassification.from_pretrained(
                                            bert_path, 
                                            output_hidden_states=True).to("cuda:0")
            self.alignment_model.eval()
        self.sent_model = sent_metric
        self.cache = AggMetricGraph.CACHE
        self.refless = refless
        self.threshold = threshold
        
        super().__init__()

    def compute_metric_single(self, complex, simplified, references):
        
        complex = complex.replace("\n", " ")
        simplified = simplified.replace("\n", " ")
        csents = sent_tokenize(complex)
        ssents = sent_tokenize(simplified)

        ref_scores = []
        for reference in references:
            key = (complex, simplified, reference) 
            if key not in self.cache:

                rsents = sent_tokenize(reference)
                rc_matrix, cr_matrix, sc_matrix, cs_matrix = get_score_matrix(
                    csents, rsents, ssents, self.alignment_model, self.tokenizer, 128
                )

                adj_list = {}
                if not self.refless:
                    consturct_graph(adj_list, cr_matrix, 'c', 'r', self.threshold)
                consturct_graph(adj_list, cs_matrix, 'c', 's', self.threshold)

                all_comps, all_cands, all_refs = [], [], []
                node_groups = bfs(adj_list)
                node_groups = merge_node_groups(node_groups)
                for group in node_groups:
                    if any(g.startswith('s') or g.startswith('c') for g in group):
                        cs = sorted([int(node[1:]) for node in group if node.startswith('c')])
                        ss = sorted([int(node[1:]) for node in group if node.startswith('s')])
                        rs = sorted([int(node[1:]) for node in group if node.startswith('r')])
                        cs = " ".join([csents[i] for i in cs])
                        ss = " ".join([ssents[i] for i in ss])
                        rs = " ".join([rsents[i] for i in rs])
                        all_comps.append(cs)
                        all_cands.append(ss)
                        all_refs.append([rs])

                if len(all_cands) == 0:
                    all_comps = [complex]
                    all_cands = [simplified]
                    all_refs = [[reference]]
                
                self.cache[key] = [all_comps, all_cands, all_refs]

            all_comps, all_cands, all_refs = self.cache[key]
            scores = self.sent_model.compute_metric(all_comps, all_cands, all_refs)
            final_score = np.mean(scores)
            ref_scores.append(final_score)

        return max(ref_scores)
    
    def compute_metric(self, complex, simplified, references) :
        scores = []
        index = 0
        for comp, simp, refs in zip(complex, simplified, references):
            print("Instance ", index, len(complex))
            scores.append(self.compute_metric_single(comp, simp, refs))
            index += 1
        return scores
    