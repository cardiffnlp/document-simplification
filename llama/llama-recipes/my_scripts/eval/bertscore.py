from bert_score import score

def calculate_bertscore(yy_, yy):
    """
    Compute BERTScore for given prediction/ground-truth pairs.
    """
    all_refs, all_cands = [], []
    for hyp, refs in zip(yy_, yy):
        for ref in refs:
            all_refs.append(ref.lower())
            all_cands.append(hyp.lower())
    
    (P, R, F), hashname = score(all_cands, all_refs, lang="en", 
                                return_hash=True, model_type="microsoft/deberta-xlarge-mnli")
    
    ind = 0
    scores = []
    for _, refs in zip(yy_, yy):
        fscores = []
        for _ in refs:
            fscores.append(P[ind].item())
            ind += 1
        scores.append(max(fscores))

    return scores

