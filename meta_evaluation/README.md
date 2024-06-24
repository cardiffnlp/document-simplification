# Evaluation of simplification/readability metrics

## Repo Structure: 
1. ```eval_datasets```: Contains the processed Cochrane, Wikipedia, OneStopQA, and perturbation datasets
with human ratings. 

2. ```metrics```: Contains wrappers for different simplification metrics, which are used to extract
correlation.

3. ``prompts``: Prompts used for llama3-based metrics.

4. ``reader_scripts``: Scripts for processing datasets.

5. ``evaluate_*.py``:  Scripts that compute correlation for different datasets.

## Calculate Correlation using Existing Metrics

The input to the correlation scripts is one of the dataset files in `eval_datasets` folder. The scripts print out the corresponding correlation values and also 
write the metric values for each instance in the `output.jsonl` file. 

### Kendall Correlation on Cochrane dataset
```
python3 evaluate_readability_kendall.py --dataset eval_datasets/cochrane_lj.jsonl --output output.jsonl 
```

### Pearson Correlation on OneStopQA dataset

```
python3 evaluate_readability_pearson.py --dataset eval_datasets/qa.jsonl --output output.jsonl 
```

### Correlation on Wikipedia dataset

```
python3 evaluate_readability_pearson.py --dataset eval_datasets/dwiki_final.jsonl --output output.jsonl 
```

### Correlation on different perturbation datasets

```
python3 evaluate_readability_kendall.py --dataset eval_datasets/peturb_deletion.jsonl --output output.jsonl 
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
2. Initialize an instance of the new metric and add it to the list of metrics in the `evaluate_readability_kendall.py`
and `evaluate_readability_pearson.py` files.
```
from metrics.mymetric import MyMetric
.....

metrics = [MyMetric()]
.....

compute_metrics(dataset, metrics)
```
