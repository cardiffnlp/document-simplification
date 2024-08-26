import textstat

class ARI:

    def __init__(self):
        # Assign a name to the metric, which will be displayed while running the eval scripts.
        self.name = "ARI" 
        # Any other initializations


    def compute_metric(self, complex, simplified, references):
        # complex: List of complex texts.
        # simplified: List of simplified texts corresponding to the complex texts.
        # references: List of list of reference texts corresponding to the complex texts.
        # There can be multiple references.
        scores = []
        # List of scores corresponding to the simplified texts
        for simp in simplified:
            scores.append(textstat.automated_readability_index(simp))

        return scores
       