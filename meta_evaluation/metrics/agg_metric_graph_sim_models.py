import torch
import bert_score
import numpy as np
from nltk import sent_tokenize
from sklearn.utils.extmath import softmax
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers.data.processors import InputExample, glue_convert_examples_to_features

from sentence_transformers import SentenceTransformer
from sentence_transformers import util


def get_score_matrix(model, csents, rsents, ssents):

    cembeddings = model.encode(csents)
    rembeddings = model.encode(rsents)
    sembeddings = model.encode(ssents)

    cr_matrix = util.pytorch_cos_sim(cembeddings, rembeddings)
    cs_matrix = util.pytorch_cos_sim(cembeddings, sembeddings)

    rc_matrix = util.pytorch_cos_sim(rembeddings, cembeddings)
    sc_matrix = util.pytorch_cos_sim(sembeddings, cembeddings)

    # all_sents_tgt = [rsents, csents, ssents, csents]
    # all_sents_cand = [csents, rsents, csents, ssents]

    # tgt_can_pairs = []
    # for tgt_sentences, can_sentences in zip(all_sents_tgt, all_sents_cand):
    #     for t in tgt_sentences:
    #         for c in can_sentences:
    #             tgt_can_pairs.append((t, c))
    # tgts, cans = list(zip(*tgt_can_pairs))
    # tgts = list(tgts)
    # cans = list(cans)
    # (_, _, pairwise_scores), _ = bert_score.score(cans, tgts, lang="en", 
    #                                     return_hash=True, verbose=True, idf=False,
    #                                     model_type="roberta-large", batch_size=64)
    # nc, ns, nr = len(csents), len(ssents), len(rsents)
    # rc_matrix = np.array(pairwise_scores[:nc * nr]).reshape((nr, nc))
    # start_ind = nc * nr
    # end_ind = start_ind + nc * nr
    # cr_matrix = np.array(pairwise_scores[start_ind: end_ind]).reshape((nc, nr))
    # start_ind = end_ind
    # end_ind = start_ind + ns * nc
    # sc_matrix = np.array(pairwise_scores[start_ind: end_ind]).reshape((ns, nc))
    # cs_matrix = np.array(pairwise_scores[-1 * nc * ns:]).reshape((nc, ns))
    return rc_matrix, cr_matrix, sc_matrix, cs_matrix


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

    # print("Old groups")
    # for group in node_groups:
    #     print(group)

    # print("New groups")
    # for group in new_node_groups:
    #     print(group)

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
            # print(key, node_group)
            node_groups.append(node_group)
    return node_groups
    

class AggMeticGraphSimModels:

    CACHE = {}
    def __init__(self, bert_path, sent_metric, refless=False):
        self.name = "Aggregation Metric Graph Sim Model SBERT -" + sent_metric.name

        self.alignment_model = SentenceTransformer("all-mpnet-base-v2")
        self.sent_model = sent_metric
        self.cache = AggMeticGraphSimModels.CACHE
        self.refless = refless
        self.threshold = 0.5
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

                # print("**"* 20)
                # print(complex)
                # print(simplified)
                # print(reference)
                # print("---" * 20)

                rsents = sent_tokenize(reference)
                # print(len(csents), len(ssents), len(rsents))
                rc_matrix, cr_matrix, sc_matrix, cs_matrix = get_score_matrix(
                    self.alignment_model, csents, rsents, ssents,
                )

                adj_list = {}
                if not self.refless:
                    # consturct_graph(adj_list, rc_matrix, 'r', 'c')
                    consturct_graph(adj_list, cr_matrix, 'c', 'r', self.threshold)
                # consturct_graph(adj_list, sc_matrix, 's', 'c')
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
                        # print(cs)
                        # print(ss)
                        # print(rs)
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
