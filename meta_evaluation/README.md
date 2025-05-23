# Evaluation of simplification/readability metrics

## Repo Structure: 
1. ```eval_datasets```: Contains the processed Cochrane, Wikipedia, OneStopQA, and perturbation datasets
with human ratings. 

2. ```metrics```: Contains wrappers for different simplification metrics, which are used to extract
correlation.

3. ``prompts``: Prompts used for llama3-based metrics.

4. ``reader_scripts``: Scripts for processing datasets.

5. ``evaluate_*.py``:  Scripts that compute correlation for different datasets.

## Installation instructions

You can install the necessary dependencies using ``pip install -r requirements.txt``

## Calculate Correlation using Existing Metrics

The input to the correlation scripts is one of the dataset files in `eval_datasets` folder. The scripts print out the corresponding correlation values and also 
write the metric values for each instance in the `output.jsonl` file. 

Please download the BERT similarity model from [here](https://drive.google.com/file/d/1I43F4OMkCvTUMtTd9Ft3P0hGiQLcFjlT/view)

### Kendall Correlation on Cochrane dataset

```
python3 evaluate_kendall.py --dataset eval_datasets/cochrance_lj.jsonl --output output.jsonl --bert BERT_wiki/
```

### Pearson Correlation on OneStopQA dataset

```
python3 evaluate_pearson.py --dataset eval_datasets/qa.jsonl --output output.jsonl --bert BERT_wiki/
```

### Pearson Correlation on Wikipedia dataset

```
python3 evaluate_pearson.py --dataset eval_datasets/dwiki_final.jsonl --output output.jsonl --bert BERT_wiki/
```

### Kendall Correlation on different perturbation datasets

```
python3 evaluate_kendall.py --dataset eval_datasets/peturb_deletion.jsonl --output output.jsonl --bert ../BERT_wiki/
```

| Peturbation dataset files      | Error name in the paper |
| ----------- | ----------- |
| peturb_deletion.jsonl      | Deletion       |
| peturb_indomain.jsonl    | In-Document hallucination |
| peturb_outdomain.jsonl    | Out-Document hallucination  |
| peturb_fluency.jsonl    | Grammar        |
| peturb_coherence_v2.jsonl    | Coherence        |
| peturb_repetition.jsonl    | Repetition        |
| peturb_copy_v2.jsonl    | Copy        |

## Add a new metric and calcuate correlation:

1. Create a new metric wrapper under `metrics` folder.
```
class MyMetric:

    def __init__(self):
        # Assign a name to the metric, which will be displayed while running the eval scripts.
        self.name = "MyMetric" 
        # Any other initializations
    
    def compute_metric(self, complex, simplified, references):
        # complex: List of complex texts.
        # simplified: List of simplified texts corresponding to the complex texts.
        # references: List of list of reference texts corresponding to the complex texts.
        # There can be multiple references.
        scores = []
        # List of scores corresponding to the simplified texts
        return scores    
       
```
2. Initialize an instance of the new metric and add it to the list of metrics in the `evaluate_kendall.py`
and `evaluate_pearson.py` files.
```
from metrics.mymetric import MyMetric
.....

metrics = [MyMetric()]
.....

compute_metrics(dataset, metrics)
```

## Citation
Please cite if you use the above resources for your research
```
@InProceedings{NAACL-2025-Maddela,
  author = 	"Maddela, Mounica and Alva-Manchego, Fernando",
  title = 	"Adapting Sentence-Level Metrics for Document-Level Simplification",
  booktitle = 	"Proceedings of the North American Association for Computational Linguistics (NAACL)",
  year = 	"2025",
}
```
