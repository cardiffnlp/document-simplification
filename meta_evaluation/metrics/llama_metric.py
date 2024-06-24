import torch


# export HF_HOME=/home/ubuntu/simplification/envs/hf_cache
# export TRANSFORMERS_CACHE=/home/ubuntu/simplification/envs/hf_cache/huggingface/hub

class LLAMA_metric:

    def __init__(self, model, tokenizer, prompt):
        self.name = "LLAMA-Eval " + prompt 
        self.prompt = open(prompt).read().strip()
        # print(self.prompt)
        self.tokenizer = tokenizer
        self.model = model

    def compute_metric(self, complex, simplified, references):

        scores = []
        for index, (complex_single, simp_single) in enumerate(zip(complex, simplified)):
            print("Instance ", index, len(complex))
            prompt = self.prompt.replace("||complex||", complex_single)
            prompt = prompt.replace("||simplification||", simp_single)

            messages = [
                {"role": "system", "content": "You are assistant trying to help the user!"},
                {"role": "user", "content": prompt},
            ]

            input_prompt = self.tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )

            # print(input_prompt)
            input_ids = self.tokenizer(input_prompt, return_tensors="pt")            
            outputs = self.model(input_ids=input_ids.input_ids, attention_mask=input_ids.attention_mask)
            logits = outputs.logits[:, -1, :]
            probabilities = torch.softmax(logits[:,  16:21], dim=-1)
            # print(probabilities)
            weighted = probabilities * torch.Tensor([1, 2, 3, 4, 5])
            # print(weighted, weighted.sum())
            scores.append(weighted.sum().item())
            
        return scores
