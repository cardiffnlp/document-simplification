import bert_score


class BERTScore:

    def __init__(self, self_flag=False, model_name="roberta-large"):
        self.name = "BERTScore-self" if self_flag else "BERTScore-ref" 
        self.model_name = model_name
        self.name = self.name + "-" + model_name
        self.self_flag = self_flag
        super().__init__()
    
    def compute_metric(self, complex, simplified, references):

        all_comps, all_cands, all_references = [], [], []
        for single_comp, single_simp, single_refs in zip(complex, simplified, references):
            for ref in single_refs:
                all_comps.append(single_comp)
                all_cands.append(single_simp)
                all_references.append(ref)

        if self.self_flag:
            (_, _, Fs), _ = bert_score.score(all_cands, all_comps, lang="en", 
                                        return_hash=True, verbose=True, idf=False,
                                        model_type=self.model_name, batch_size=4)
        else:
            (_, _, Fs), _ = bert_score.score(all_cands, all_references, lang="en", 
                                        return_hash=True, verbose=True, idf=False,
                                        model_type=self.model_name, batch_size=4)
        
        scores = []
        ind = 0
        for single_refs in zip(references):
            bscores = []
            for _ in single_refs:
                bscores.append(Fs[ind].item())
                ind += 1
            scores.append(max(bscores) * 100.0)

        assert len(scores) == len(complex) == len(simplified) == len(references)
        return scores