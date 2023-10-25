
#!/bin/bash

# ## Coedit XL
# python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_ood.json \
#     --model grammarly/coedit-xl --output ../../experiments/swipe/outputs/paragraph/coedit-xl-0shot-ood.json \
#     --coedit_flag --instruct_flag --zero_shot_flag

# python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_id.json \
#     --model grammarly/coedit-xl --output ../../experiments/swipe/outputs/paragraph/coedit-xl-0shot-id.json \
#     --coedit_flag --instruct_flag --zero_shot_flag


# python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_ood.json \
#     --model grammarly/coedit-xl --output ../../experiments/swipe/outputs/paragraph/coedit-xl-1shot-ood.json \
#     --coedit_flag --instruct_flag 

# python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_id.json \
#     --model grammarly/coedit-xl --output ../../experiments/swipe/outputs/paragraph/coedit-xl-1shot-id.json \
#     --coedit_flag --instruct_flag


# ## Coedit Large
# python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_ood.json \
#     --model grammarly/coedit-large --output ../../experiments/swipe/outputs/paragraph/coedit-large-0shot-ood.json \
#     --coedit_flag --instruct_flag --zero_shot_flag

# python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_id.json \
#     --model grammarly/coedit-large --output ../../experiments/swipe/outputs/paragraph/coedit-large-0shot-id.json \
#     --coedit_flag --instruct_flag --zero_shot_flag


# python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_ood.json \
#     --model grammarly/coedit-large --output ../../experiments/swipe/outputs/paragraph/coedit-large-1shot-ood.json \
#     --coedit_flag --instruct_flag 

# python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_id.json \
#     --model grammarly/coedit-large --output ../../experiments/swipe/outputs/paragraph/coedit-large-1shot-id.json \
#     --coedit_flag --instruct_flag


## Coedit XL-C
python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_ood.json \
    --model grammarly/coedit-xl-composite --output ../../experiments/swipe/outputs/paragraph/coedit-xl-c-0shot-ood.json \
    --coedit_flag --instruct_flag --zero_shot_flag

python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_id.json \
    --model grammarly/coedit-xl-composite --output ../../experiments/swipe/outputs/paragraph/coedit-xl-c-0shot-id.json \
    --coedit_flag --instruct_flag --zero_shot_flag


python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_ood.json \
    --model grammarly/coedit-xl-composite --output ../../experiments/swipe/outputs/paragraph/coedit-xl-c-1shot-ood.json \
    --coedit_flag --instruct_flag 

python3 bulk_inference.py --input ../../experiments/swipe/swipe_test_id.json \
    --model grammarly/coedit-xl-composite --output ../../experiments/swipe/outputs/paragraph/coedit-xl-c-1shot-id.json \
    --coedit_flag --instruct_flag