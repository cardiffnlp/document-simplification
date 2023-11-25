import glob
import json, argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    args = parser.parse_args()

    for fpath in glob.glob(args.input):
        print(fpath)

        metrics = ["BERTScore", "QuestEval", "SMARTEval", "FKGL", "D-SARI", "S-SARI"]
        data = json.load(open(fpath))
        for metric in metrics:
            all_metric_values = []
            for instance in data:
                all_metric_values.append(instance[metric])
            print(metric, np.mean(all_metric_values, axis=0))

        cefr_labels = ["A1", "A2", "B1", "B2", "C1", "C2", "none"]
        metrics = ['CEFR_input', 'CEFR_output']
        for metric in metrics:
            all_scores = []
            for instance in data:
                score = [instance[metric].get(label, 0) for label in cefr_labels]
                all_scores.append(score)
            print(metric, np.mean(all_scores, axis=0).tolist())

        metrics = ['CEFR_ref']
        for metric in metrics:
            num_ref = len(data[0][metric])
            for i in range(num_ref):
                all_scores = []
                for instance in data:
                    score = [instance[metric][i].get(label, 0) for label in cefr_labels]
                    all_scores.append(score)
                print(metric, i, np.mean(all_scores, axis=0).tolist())

