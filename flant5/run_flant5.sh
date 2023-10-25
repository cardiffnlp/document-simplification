
#!/bin/bash

## Flan T5 XL
python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_ood.json \
    --model google/flan-t5-xl --output ../../experiments/swipe/outputs/paragraph/flant5-xl-0shot-ood.json \
    --instruct_flag --zero_shot_flag

python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_id.json \
    --model google/flan-t5-xl --output ../../experiments/swipe/outputs/paragraph/flant5-xl-0shot-id.json \
    --instruct_flag --zero_shot_flag


python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_ood.json \
    --model google/flan-t5-xl --output ../../experiments/swipe/outputs/paragraph/flant5-xl-1shot-ood.json \
    --instruct_flag 

python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_id.json \
    --model google/flan-t5-xl --output ../../experiments/swipe/outputs/paragraph/flant5-xl-1shot-id.json \
    --instruct_flag


## Flan T5 Large
python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_ood.json \
    --model google/flan-t5-large --output ../../experiments/swipe/outputs/paragraph/flant5-large-0shot-ood.json \
    --instruct_flag --zero_shot_flag

python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_id.json \
    --model google/flan-t5-large --output ../../experiments/swipe/outputs/paragraph/flant5-large-0shot-id.json \
    --instruct_flag --zero_shot_flag


python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_ood.json \
    --model google/flan-t5-large --output ../../experiments/swipe/outputs/paragraph/flant5-large-1shot-ood.json \
    --instruct_flag 

python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_id.json \
    --model google/flan-t5-large --output ../../experiments/swipe/outputs/paragraph/flant5-large-1shot-id.json \
    --instruct_flag
