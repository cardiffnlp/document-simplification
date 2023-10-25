# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

from datasets import load_dataset

from llama_recipes.datasets.utils import Concatenator

def get_preprocessed_swipe(dataset_config, tokenizer, split):

    dataset = load_dataset('json', data_files={'train': '/home/ubuntu/simplfication/experiments/swipe/swipe_train.json', 
                                               'validation': '/home/ubuntu/simplfication/experiments/swipe/swipe_val.json'},
                                               split=split)
    # dataset = load_dataset('json', data_files={'train': '/home/ubuntu/simplfication/experiments/newsela_auto_llama/0-3-paragraph/train.json', 
                                            #    'validation': '/home/ubuntu/simplfication/experiments/newsela_auto_llama/0-3-paragraph/valid.json'},
                                            #    split=split)
    instruct_flag = False
    if instruct_flag:
        prompt = (f"{{bos_token}} [INST] <<SYS>> You are a helpful, respectful and honest assistant. Please rewrite the following text into simpler language that is easier to understand while retaining the original meaning. <</SYS>>\n\n{{input}} [/INST]\n{{output}}{{eos_token}}")
    else:
        prompt = (f"{{bos_token}}Rewrite the following text into simpler language that is easier to understand while retaining the original meaning:\n{{input}}\n---\nSimplified text:\n{{output}}{{eos_token}}")


    def apply_prompt_template(sample):
        return {
            "text": prompt.format(
                input=sample["r_content"],
                output=sample["s_content"],
                eos_token=tokenizer.eos_token,
                bos_token=tokenizer.bos_token
            )
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    print(dataset)
    print(dataset[0]['text'])
    print(dataset[15]['text'])
     
    dataset = dataset.map(
        lambda sample: tokenizer(sample["text"]),
        batched=True,
        remove_columns=list(dataset.features),
    ).map(Concatenator(), batched=True)
    return dataset
