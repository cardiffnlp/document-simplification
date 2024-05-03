# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SMART scorer."""

from nltk import tokenize
import numpy as np

from metrics.bleu import BLEU

class MeanAggregateFunction():
  def __call__(self, scores, **kwargs):
    return np.mean(scores)
  
class SmartScorer:
  """Calculate SMART scores between two texts.

  Sample usage:
  1. When a matching function is provided, you can use the smart_score() fn:
    scorer = SmartScorer(matching_fn=matcher)
    score = scorer.smart_score(
        'This is the first sentence. This one is the second.',
        'The first sentence is this. This follows the first.')

  2. When a matching function is not provided, you can use the
       smart_score_precomputed() fn providing a score_matrix which contains the
       scores for all pairs of sentences in the reference and candidate texts.
       This one is particularly useful with model-based matchers such as
       BLEURT, ANLI, etc.:
    scorer = SmartScorer(matching_fn=None)
    score = scorer.smart_score_precomputed(
        score_matrix=[[0.9, 0.3],[0.2, 0.4]])
  """

  CACHE = {}
  
  def __init__(
      self,
      matching_fn = BLEU(),
      split_fn = tokenize.sent_tokenize,
      aggregate_fn = MeanAggregateFunction(),
      is_symmetric_matching = True,
      final_smart_type = "smartL",
      refless = False
  ):
    """Initializes a new SmartScorer.

    Valid SMART types that can be computed are:
      smart1: Unigram-based scoring.
      smart2: Bigram-based scoring.
      smartL: Longest common subsequence based scoring.
    Note that the unit of ngrams here are chunks of tokens (sentences by
      default). This is different from the token-level ngrams used in standard
      ROUGE.

    The matching_fn should accept two lists of strings, the reference and
      the candidate texts.

    Args:
      smart_types: A list of SMART types to calculate.
      matching_fn: Function used to match sentences. If None, a score matrix
        will be required by the smart_score function.
      split_fn: Function used to split the text. The default is to use the
        sentence splitter in NLTK.
      aggregate_fn: Function used to aggregate sentence-level scores.
      is_symmetric_matching: if True, matching_fn(i,j) = matching_fn(j,i).
    """

    self.smart_types = ['smart1', 'smart2', 'smartL']
    self.matching_fn = matching_fn
    self.split_fn = split_fn
    self.aggregate_fn = aggregate_fn
    self.is_symmetric_matching = is_symmetric_matching
    self.final_smart_type = final_smart_type
    self.refless = refless
    self.cache = SmartScorer.CACHE

    self.name = " - ".join(["Smart Scorer", final_smart_type, matching_fn.name])

  def _smart_1(
      self,
      score_matrix,
      rev_score_matrix,
  ):
    """Calculate SMART-1 score."""
    if isinstance(score_matrix, list):
      score_matrix = np.array(score_matrix)
    if isinstance(rev_score_matrix, list):
      rev_score_matrix = np.array(rev_score_matrix)

    recall = self.aggregate_fn(np.max(score_matrix, 1))
    precision = self.aggregate_fn(np.max(rev_score_matrix, 1))
    f = (
        (2 * precision * recall) / (precision + recall)
        if precision + recall != 0
        else 0.0
    )
    return {'recall': recall, 'precision': precision, 'fmeasure': f}

  def _smart_2(
      self,
      score_matrix,
      rev_score_matrix,
  ):
    """Calculate SMART-2 score."""
    # Pad matrix with zeros in for cases where there is only one sentence.
    # This changes the algorithm (i.e., using R-1 for one-sentence summaries)
    # but improves overall correlation.
    score_matrix = np.pad(score_matrix, [(1, 1), (1, 1)])
    rev_score_matrix = np.pad(rev_score_matrix, [(1, 1), (1, 1)])
    # Calculate bigram scores.
    bigram_score_matrix = (score_matrix[:-1, :-1] + score_matrix[1:, 1:]) / 2.0
    rev_bigram_score_matrix = (
        rev_score_matrix[:-1, :-1] + rev_score_matrix[1:, 1:]
    ) / 2.0

    recall = self.aggregate_fn(np.max(bigram_score_matrix, 1))
    precision = self.aggregate_fn(np.max(rev_bigram_score_matrix, 1))
    f = (
        (2 * precision * recall) / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {'recall': recall, 'precision': precision, 'fmeasure': f}

  def _smart_l(
      self,
      score_matrix,
      rev_score_matrix,
  ):
    """Calculate SMART-L score."""
    if isinstance(score_matrix, list):
      score_matrix = np.array(score_matrix)
    row_len, col_len = score_matrix.shape

    recall = self._soft_lcs(score_matrix) / row_len
    precision = self._soft_lcs(rev_score_matrix) / col_len
    f = (2 * precision * recall) / (precision +
                                    recall) if precision + recall > 0 else 0.0
    return {'recall': recall, 'precision': precision, 'fmeasure': f}

  def _soft_lcs(self, score_matrix):
    """Soft-LCS algorithm.

    Two main differences with the regular LCS:
    1. We allow "soft matching"; score doesn't need to be 1 to be considered
       match.
    2. We allow multiple matching; a sentence can be matched to more than 1
       sentences as long as they are consecutive.
    More information here: https://arxiv.org/pdf/2208.01030.pdf

    Args:
      score_matrix: The scores for each pair of sentences.

    Returns:
      The soft-LCS score.
    """
    row_len, col_len = score_matrix.shape

    # dp(0, 0) = 0
    # dp(i, j) = max(dp(i-1, j-1) + s(i, j), dp(i-1, j) + s(i, j), dp(i, j-1))
    dp_table = [[0 for _ in range(col_len + 1)] for _ in range(row_len + 1)]
    for i in range(row_len + 1):
      for j in range(col_len + 1):
        if i != 0 and j != 0:
          dp_table[i][j] = max(
              dp_table[i - 1][j - 1] +
              score_matrix[i - 1, j - 1],  # Match i and j.
              dp_table[i - 1][j] +
              score_matrix[i - 1, j - 1],  # Re-match i-1 and j.
              dp_table[i][j - 1])  # Skip j.
    return dp_table[row_len][col_len]

  def smart_score_precomputed(
      self,
      ref_score_matrix,
      src_score_matrix = None,
      rev_ref_score_matrix = None,
      rev_src_score_matrix = None,
  ):
    """Calculates SMART scores given a precomputed score matrix.

    Args:
      ref_score_matrix: The pre-calculated scores, where ref_score_matrix[i][j]
        is the score between reference sentence i and candidate sentence j.
      src_score_matrix: The pre-calculated scores, where src_score_matrix[i][j]
        is the score between source sentence i and candidate sentence j.
      rev_ref_score_matrix: If the scorer is assymetric, the pre-calculated
        scores between candidate sentence j and reference sentence i.
      rev_src_score_matrix: If the scorer is assymetric, the pre-calculated
        scores between candidate sentence j and source sentence i.

    Returns:
      A mapping of each SMART type to its scores.
    """
    return_dict = {}

    if not self.is_symmetric_matching and rev_ref_score_matrix is None:
      # Reverse matrices are required.
      raise ValueError(
          'Reverse matrices are required for symmetric matching. Ensure that'
          ' you set both `rev_ref_score_matrix` and `rev_src_score_matrix`, or'
          ' set `is_symmetric_matching` to True.'
      )
    if src_score_matrix is not None:
      if not self.is_symmetric_matching and rev_src_score_matrix is None:
        raise ValueError(
            'Reverse matrices are required for symmetric matching. Ensure that'
            ' you set both `rev_ref_score_matrix` and `rev_src_score_matrix`,'
            ' or set `is_symmetric_matching` to True.'
        )

    if rev_ref_score_matrix is None:
      rev_ref_score_matrix = np.transpose(ref_score_matrix)
    if rev_src_score_matrix is None and src_score_matrix is not None:
      # Only required if src_score_matrix is available.
      rev_src_score_matrix = np.transpose(src_score_matrix)

    smart_fn_dict = {
        'smart1': self._smart_1,
        'smart2': self._smart_2,
        'smartL': self._smart_l
    }

    for smart_type in self.smart_types:

      smart_fn = smart_fn_dict[smart_type]

      ref_smart = smart_fn(ref_score_matrix, rev_ref_score_matrix)
      if src_score_matrix is not None:
        src_smart = smart_fn(src_score_matrix, rev_src_score_matrix)
        return_dict[smart_type] = {
            x: max(ref_smart[x], src_smart[x])
            for x in ['precision', 'recall', 'fmeasure']
        }
      else:
        return_dict[smart_type] = ref_smart

    return return_dict

  def _get_score_matrix(self, tgt_sentences,
                        can_sentences):
    """Gets the score matrix using the given matching_fn.

    Args:
      tgt_sentences: The list of target sentences, which can either be from the
        source or the reference.
      can_sentences: The list of candidate sentences

    Returns:
      A matrix containing pairwise scores.
    """
    tgt_can_pairs = []
    for t in tgt_sentences:
      for c in can_sentences:
        tgt_can_pairs.append((t, c))
    tgts, cans = list(zip(*tgt_can_pairs))
    tgts = [[tgt] for tgt in list(tgts)]
    cans = list(cans)
    comps = ["" for _ in cans]
    # We assume that the matching function can be asymmetrical, i.e.,
    # interchanging the position of r and c may return a different score.
    pairwise_scores = self.matching_fn.compute_metric(comps, cans, tgts)
    score_matrix = np.array(pairwise_scores).reshape(
        (len(tgt_sentences), len(can_sentences)))
    return score_matrix

  def smart_score(
      self,
      reference,
      candidate,
      source = None):
    """Calculates SMART scores given two sets of sentences.

    Currently, this can only work on examples with only one reference summary.

    Args:
      reference: The reference text.
      candidate: The candidate text.
      source: The source text. If provided, it will return an aggregated version
        of SMART using both source and reference.

    Returns:
      A mapping of each SMART type to its scores.
    """

    key_cache = (candidate, reference, self.matching_fn.name)
    if key_cache in self.cache:
      return self.cache[key_cache][self.final_smart_type]['fmeasure']

    # Split reference/candidate into sentences if necessary.
    src_sentences = None
    if isinstance(reference, str):
      ref_sentences = self.split_fn(reference)
      can_sentences = self.split_fn(candidate)
      if source is not None:
        src_sentences = self.split_fn(source)
    else:
      ref_sentences = reference
      can_sentences = candidate
      if source is not None:
        src_sentences = source
    
    # Calculate pairwise matching scores between sentences in ref and can.
    if self.matching_fn is None:
      raise NotImplementedError('A matching function should be implemented.')
    ref_score_matrix = self._get_score_matrix(ref_sentences, can_sentences)
    if self.is_symmetric_matching:
      rev_ref_score_matrix = np.transpose(ref_score_matrix)
    else:
      rev_ref_score_matrix = self._get_score_matrix(
          can_sentences, ref_sentences
      )
    if source is not None:
      src_score_matrix = self._get_score_matrix(src_sentences, can_sentences)
      if self.is_symmetric_matching:
        rev_src_score_matrix = np.transpose(src_score_matrix)
      else:
        rev_src_score_matrix = self._get_score_matrix(
            can_sentences, src_sentences
        )
    else:
      src_score_matrix = None
      rev_src_score_matrix = None

    metric_values = self.smart_score_precomputed(
        ref_score_matrix,
        src_score_matrix,
        rev_ref_score_matrix,
        rev_src_score_matrix,
    )
    self.cache[key_cache] = metric_values
    return metric_values[self.final_smart_type]['fmeasure']

  def compute_metric(self, complex, simplified, references) :
        scores = []
        index = 0
        for comp, simp, refs in zip(complex, simplified, references):
            print("Instance ", index, len(complex))
            if self.refless:
              score = self.smart_score(comp, simp)
            else:
              ref_scores = []
              for ref in refs: 
                ref_score = self.smart_score(ref, simp)
                ref_scores.append(ref_score)
              score = np.max(ref_scores)
            scores.append(score)
            index += 1
        return scores
