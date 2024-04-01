
from metrics.matching_functions import BleuMatchingFunction
from smart_eval import scorer

class SMART:

    name = "SMART"

    def __init__(self, matcher):
        self.metric = scorer.SmartScorer(matching_fn=matcher)

    def compute_metric(self, complex, simplified, references):
        
        scores = []
        for reference in references:
            # print(simplified)
            # print(reference)
            scores.append(self.metric.smart_score(complex, simplified, reference)['smartL']['fmeasure'])
        return max(scores)
