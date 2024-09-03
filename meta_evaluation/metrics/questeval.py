
from questeval.questeval_metric import QuestEval

class QuestEvalMetric:

    name = "QuestEval"

    def __init__(self):
        self.questeval = QuestEval(
            task='text_simplification',
            no_cuda=True,
            qg_batch_size=16,
            clf_batch_size=16)

    def compute_metric(self, complex, simplified, references):
        scores = self.questeval.corpus_questeval(
                            hypothesis=simplified, 
                            sources=complex,
                            list_references=references,
                            batch_size=16
                )
        return scores['ex_level_scores']
