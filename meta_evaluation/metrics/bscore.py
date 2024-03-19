import bert_score


class BERTScore:

    def __init__(self, self_flag=False, model_name="roberta-large"):
        self.name = "BERTScore-self" if self_flag else "BERTScore-ref" 
        self.model_name = model_name
        self.name = self.name + "-" + model_name
        self.self_flag = self_flag
        super().__init__()

    def compute_metric(self, complex, simplified, references):

        all_comps, all_cands = [], []
        for _ in references:
            all_cands.append(simplified)
            all_comps.append(complex)

        if self.self_flag:
            (_, _, Fs), _ = bert_score.score(all_cands, all_comps, lang="en", 
                                        return_hash=True, verbose=True, idf=False,
                                        model_type=self.model_name)
        else:
            (_, _, Fs), _ = bert_score.score(all_cands, references, lang="en", 
                                        return_hash=True, verbose=True, idf=False,
                                        model_type=self.model_name)
        
    
        print(Fs, max(Fs))
        return max(Fs).item() * 100.0

