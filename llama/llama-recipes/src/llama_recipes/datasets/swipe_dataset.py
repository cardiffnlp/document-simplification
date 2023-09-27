# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

from datasets import load_dataset

from llama_recipes.datasets.utils import Concatenator

def get_preprocessed_swipe(dataset_config, tokenizer, split):


    print(split)
    dataset = load_dataset('json', data_files={'train': '/home/ubuntu/simplfication/experiments/swipe/swipe_train.json', 
                                               'test': '/home/ubuntu/simplfication/experiments/swipe/swipe_val.json',
                                               'validation': '/home/ubuntu/simplfication/experiments/swipe/swipe_test_id.json'},
                                               split=split)

    prompt = (f"Simplify the following text:\n{{input}}\n---\nSimplified text:\n{{output}}{{eos_token}}")

    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                input=sample["r_content"],
                output=sample["s_content"],
                eos_token=tokenizer.eos_token,
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    print(dataset)
    print(dataset[0])
     
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset
