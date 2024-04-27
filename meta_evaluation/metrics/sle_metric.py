import sys

sys.path.append('/Users/mmaddela3/Documents/simplification_evaluation/external_repos/sle') 

from sle.scorer import SLEScorer


class SLE_metric:

    def __init__(self, diff=False):
        self.name = "SLE-diff" if diff else "SLE" 
        self.diff = diff
        self.sle_metric = SLEScorer("liamcripwell/sle-base", "cpu")
    
    def compute_metric(self, complex, simplified, references):        
        if self.diff:
            scores = self.sle_metric.score(simplified)
            return [sc * -100.0 for sc in scores['sle']]
        else:
            scores = self.sle_metric.score(simplified, inputs=complex)
            return [sc * -100.0 for sc in scores['sle']]
