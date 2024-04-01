# import sys
# from lens.lens_score import LENS

# model_path = sys.argv[1]
# # Original LENS is a real-valued number. 
# # Rescaled version (rescale=True) rescales LENS between 0 and 100 for better interpretability. 
# # You can also use the original version using rescale=False
# metric = LENS(model_path, rescale=True)

# complex = ["They are culturally akin to the coastal peoples of Papua New Guinea."]
# simple = ["They are culturally similar to the people of Papua New Guinea."]
# references = [
#     [
#         "They are culturally similar to the coastal peoples of Papua New Guinea.",
#         "They are similar to the Papua New Guinea people living on the coast."
#     ]
# ]

# scores = metric.score(complex, simple, references, batch_size=1, gpus=0)
# print(scores)

# import sys

# sys.path.append('/Users/mmaddela3/Documents/simplification_evaluation/external_repos/sle') 

# from sle.scorer import SLEScorer

# scorer = SLEScorer("liamcripwell/sle-base", "cpu")

# texts = [
#   "Here is a simple sentence.",
#   "Here is an additional sentence that makes use of more complex terminology."
# ]

# # raw simplicity estimates
# results = scorer.score(texts)
# print(results) # {'sle': [3.9842946529388428, 0.5840105414390564]}

# # delta from input sentences
# results = scorer.score([texts[0]], inputs=[texts[1]])
# print(results) # {'sle': [3.9842941761016846], 'sle_delta': [3.4002838730812073]}


import sys

sys.path.append('/Users/mmaddela3/Documents/simplification_evaluation/external_repos/google-research') 

from smart_eval import matching_functions
from smart_eval import scorer

complex = ["They are culturally akin to the coastal peoples of Papua New Guinea. This is a test."]
simple = ["They are culturally similar to the people of Papua New Guinea. I know, this is a test."]
references = [
    [
        "They are culturally similar to the coastal peoples of Papua New Guinea.",
        "They are similar to the Papua New Guinea people living on the coast."
    ]
]

matcher = matching_functions.BleuMatchingFunction()
smart_scorer = scorer.SmartScorer(matching_fn=matcher)
score = smart_scorer.smart_score(complex[0], simple[0])
print(score)
