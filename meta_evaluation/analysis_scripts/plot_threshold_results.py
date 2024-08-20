import sys
import glob
import json

import matplotlib.pyplot as plt
from utils import pairwise_kendall, pointwise_pearson

results_path = sys.argv[1]

plt.figure(figsize=(10,6))
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

for dataset_path, ax in zip(["cochrane", "dwiki", "qa"], [ax1, ax2, ax3]):
    ind_metrics = {}
    for fpath in glob.glob(dataset_path + "/*"):
        data = json.load(open(fpath))
        threshold = fpath.replace(".jsonl", "")[-3:]
        ind_metrics[threshold] = data

    metric_names = ['LENS', "SARI", "BERTScore"]
    thresholds = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
    metrics = ["Aggregation Metric Graph -LENS", "Aggregation Metric Graph -SARI","Aggregation Metric Graph -BERTScore-ref-roberta-large"]
    markers = ["*", "+", "o"]
    for metric, marker, metric_name in zip(metrics, markers, metric_names):
        xs, ys = [], []
        for threshold in thresholds:
            # print(threshold, pairwise_kendall(ind_metrics[threshold], metric))
            # print(threshold, pointwise_pearson(ind_metrics[threshold], metric, ["meaning", "grammar", "simplicity-overall"]))
            print(threshold, pointwise_pearson(ind_metrics[threshold], metric))
            print()
            xs.append(float(threshold))
            ys.append(pointwise_pearson(ind_metrics[threshold], metric))
        ax.plot(xs, ys, marker=marker, label=metric_name)
        plt.xticks([i * 0.1 for i in range(0, 10)])
    
    # plt.grid(True, color = "grey", linewidth = "1",  axis = 'y')
# plt.legend(, fontsize="15") 
    # ax.ylim(0.0, 0.6)
# ax.legend(bbox_to_anchor=(1.05, 0), loc='upper center', borderaxespad=0.)
ax.legend()
# fig.tight_layout()
plt.show()
plt.savefig("test.png")
    