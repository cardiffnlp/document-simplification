import glob
import json
import torch
import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer


from llama_recipes.inference.model_utils import load_peft_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--cefr", type=str)
    parser.add_argument('--instruct_flag', action='store_true')
    args = parser.parse_args()

    model_id = args.model       
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
    if args.model_dir:
        model = load_peft_model(model, args.model_dir)
    model.eval()
    
    index = 0
    output_jsons = []
    data = json.load(open(args.input))
    for json_instance in data:
        input_ = json_instance["r_content"]
        print(index)

        # if len(input_) > 10000:
        #     output = {
        #         "input": input_,
        #         "prompt": "",
        #         "output": "",
        #         "target": json_instance["s_content"],
        #     }
        #     output_jsons.append(output)
        #     index += 1
        #     continue

        if args.instruct_flag:
            eval_prompt =  "<s>[INST] <<SYS>>You are a helpful, respectful and honest assistant. Please rewrite the following text at " + args.cefr + " level." + \
            " <</SYS>>\n" + input_ + "[/INST]\n"
        else:
            eval_prompt = "<s>Rewrite the following text into simpler language that is easier to understand while retaining the original meaning:\n" + input_ + "\n---\nSimplified text:\n"

        model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            generated_text = model.generate(**model_input,
                                                max_new_tokens=2048,
                                                do_sample=True,
                                                repetition_penalty=1.01,
                                                top_k=50)[0]

            generated_text = tokenizer.decode(generated_text, skip_special_tokens=True)
            
            
            output_text = generated_text.split("[/INST]")[1].strip() if args.instruct_flag else \
                            generated_text.split("---\nSimplified text:")[1].strip()
            output = {
                "input": input_,
                "prompt": eval_prompt,
                "generated": generated_text,
                "output": output_text,
                "target": json_instance["s_content"],
            }
            output_jsons.append(output)
            print(index, generated_text)
            print("********")
            index += 1
            # if index == 2:
            #     break

    with open(args.output, "w") as fp:
        json.dump(output_jsons, fp)


                        


                




