"""
Implements the evaluation metrics based on BLEU score
"""

import numpy as np
from typing import List

import sacrebleu

from nltk import word_tokenize


def normalize(string, lowercase):
    return string.lower() if lowercase else string


def corpus_bleu(
    sys_sents: List[str],
    refs_sents: List[List[str]],
    lowercase: bool = False,
):
    sys_sents = [normalize(sent, lowercase) for sent in sys_sents]
    refs_sents = [[normalize(sent, lowercase) for sent in ref_sents] for ref_sents in refs_sents]

    bleu_scorer = sacrebleu.metrics.BLEU()

    return bleu_scorer.corpus_score(
        sys_sents,
        refs_sents,
    ).score


def sentence_bleu(
    sys_sent: str,
    ref_sents: List[str],
    lowercase: bool = False
):

    return corpus_bleu(
        [sys_sent],
        [[ref] for ref in ref_sents],
        lowercase=lowercase,
    )

class BLEU:

    name = "BLEU"
    
    def compute_metric(self, complex, simplified, references):
        scores = []
        for single_simp, single_refs in zip(simplified, references):
            score = sentence_bleu(single_simp, single_refs, lowercase=True)
            scores.append(score)
        return scores