import nltk
import json
import argparse
import numpy as np
from easse.bleu import corpus_bleu, sentence_bleu
from easse.fkgl import corpus_fkgl

from eval.dsari import compute_dsari
from eval.ssari import compute_sari
from cefr_classifier import get_cefr_levels, get_cefr_rules
from eval.bertscore import calculate_bertscore
from eval.questeval import calculate_questeval
from eval.smart_eval import matching_functions, scorer


# TODO: Incoporate LENS into SMARTEval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", type=str)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    
    data = json.load(open(args.input))

    dsari_all = []
    ssari_all = []
    fkgl_all = []
    sbleu_all = []
    all_cefr_scores = []
    input_all, output_all, target_all = [], [], []

    cefr_rules = get_cefr_rules(args.html)
    print("Size", len(data))

    for instance in data:
        input = instance['input']
        output = instance['output']
        target = instance['target']

        output = output.split("\n")
        output = " ".join([para for para in output if len(para) > 0])

        dsari_instance = compute_dsari(input, output, target)
        dsari_all.append(dsari_instance)

        ssari_instance = compute_sari(input, output, target)
        ssari_all.append(ssari_instance)

        out_doc_sents = nltk.sent_tokenize(output)
        fkgl_instance = corpus_fkgl(out_doc_sents)
        fkgl_all.append(fkgl_instance)

        sbleu = sentence_bleu(output, target)
        sbleu_all.append(sbleu)

        cefr_input_score = get_cefr_levels(cefr_rules, instance['input'])
        cefr_output_score = get_cefr_levels(cefr_rules, instance['output'])
        cefr_ref_scores = [get_cefr_levels(cefr_rules, target) for target in instance['target']]
        all_cefr_scores.append([cefr_output_score[label] for label in ["A1", "A2", "B1", "B2", "C1", "C2", "none"]])

        instance['S-BLEU'] = sbleu
        instance['FKGL'] = fkgl_instance
        instance['D-SARI'] = dsari_instance
        instance['S-SARI'] = ssari_instance
        instance['CEFR_input'] = cefr_input_score
        instance['CEFR_output'] = cefr_output_score
        instance['CEFR_ref'] = cefr_ref_scores

        input_all.append(input)
        output_all.append(output)
        target_all.append(target)


    print("D-SARI", np.mean(dsari_all, axis=0) )
    print("S-SARI", np.mean(ssari_all, axis=0))
    print("FKGL", np.mean(fkgl_all, axis=0))
    print("BLEU", corpus_bleu(output_all, target_all, lowercase=True))

    bscore_all = calculate_bertscore(output_all, target_all)
    print("BERTScore", np.mean(bscore_all))

    questeval_all = calculate_questeval(input_all, output_all)
    print("QuestEval", np.mean(questeval_all)) 

    scorer_final = scorer.SmartScorer()
    smarts_all = []
    for output, target in zip(output_all, target_all):
        smarts = scorer_final.smart_score(target, output)["smartL"]
        smarts_all.append([smarts["precision"], smarts["recall"], smarts["fmeasure"]])
    print("SMARTEval", np.mean(smarts_all, axis=0))

    corpus_cefr = np.sum(all_cefr_scores, axis=0) * 100.0 / np.sum(all_cefr_scores)
    for bval, qval, sval, instance in zip(bscore_all, questeval_all, smarts_all, data):
        instance['BERTScore'] = bval
        instance['QuestEval'] = qval
        instance['SMARTEval'] = sval
    print("CEFR Scores", corpus_cefr)

    print(data[0])
    with open(args.output, "w") as fp:
        json.dump(data, fp)
