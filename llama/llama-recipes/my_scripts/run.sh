#! /bin/bash

# INPUT=$1
# MODEL=$2
# OUTPUT=$3

# python3 my_scripts/bulk_inference.py --input $INPUT \
#     --instruct_flag --model $MODEL \
#     --output $OUTPUT-A1.json --cefr A1

# python3 my_scripts/bulk_inference.py --input $INPUT \
#     --instruct_flag --model $MODEL \
#     --output $OUTPUT-A2.json --cefr A2

# python3 my_scripts/bulk_inference.py --input $INPUT \
#     --instruct_flag --model $MODEL \
#     --output $OUTPUT-B1.json --cefr B1

# python3 my_scripts/bulk_inference.py --input $INPUT \
#     --instruct_flag --model $MODEL \
#     --output $OUTPUT-B2.json --cefr B2

# python3 my_scripts/bulk_inference.py --input $INPUT \
#     --instruct_flag --model $MODEL \
#     --output $OUTPUT-C1.json --cefr C1

# python3 my_scripts/bulk_inference.py --input $INPUT \
#     --instruct_flag --model $MODEL \
#     --output $OUTPUT-C2.json --cefr C2

INPUT=$1
python3 calcluate_metrics.py --input $INPUT-A1.json \
    --html ../../../../experiments/CEFR/vocab.html \
    --output $INPUT-A1-scores.json

python3 calcluate_metrics.py --input $INPUT-A2.json \
    --html ../../../../experiments/CEFR/vocab.html \
    --output $INPUT-A2-scores.json

python3 calcluate_metrics.py --input $INPUT-B1.json \
    --html ../../../../experiments/CEFR/vocab.html \
    --output $INPUT-B1-scores.json

python3 calcluate_metrics.py --input $INPUT-B2.json \
    --html ../../../../experiments/CEFR/vocab.html \
    --output $INPUT-B2-scores.json

python3 calcluate_metrics.py --input $INPUT-C1.json \
    --html ../../../../experiments/CEFR/vocab.html \
    --output $INPUT-C1-scores.json

python3 calcluate_metrics.py --input $INPUT-C2.json \
    --html ../../../../experiments/CEFR/vocab.html \
    --output $INPUT-C2-scores.json
