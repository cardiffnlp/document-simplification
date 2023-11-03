from questeval.questeval_metric import QuestEval

def calculate_questeval(xx, yy_, batch_size=4):
    """
    Compute BERTScore-based QuestEval for given source/prediction pairs.
    """
    questeval = QuestEval(
        list_scores=('answerability', 'bertscore',),
        limit_sent=None,
        task="text_simplification", 
    )

    score = questeval.corpus_questeval(
        hypothesis=yy_,
        sources=xx,
        batch_size=batch_size,
    )
    return score["ex_level_scores"]