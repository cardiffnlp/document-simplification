import glob
import json
import torch
import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer


from llama_recipes.inference.model_utils import load_model, load_peft_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument("input", type=str)
    # parser.add_argument("model_dir", type=str)
    # parser.add_argument("output", type=str)
    args = parser.parse_args()

    model_id="meta-llama/Llama-2-7b-chat-hf"
    # model_id=args.model_id        
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
    # model = load_peft_model(model, args.model_dir)
    model.eval()

    eval_prompt = "[INST] <<SYS>>You are a helpful, respectful and honest assistant. Please rewrite the following medical abstract into a plain summary. " + \
        "Please stay faithful to the input text. <</SYS>>\n" + \
        "Analysis showed a higher rate of weight gain in the high-volume feeds group: mean difference 6.20 g/kg/d (95% confidence interval 2.71 to 9.69). There " + \
        "was no increase in the risk of feed intolerance or necrotising enterocolitis with high-volume feeds, but 95% confidence intervals around these estimates were wide. [/INST]\n"
        # " Transport, or transportation (as it is called in the United States), is the movement of people and goods from one " + \
        # "place to another. The term is derived from the Latin trans meaning across and portare meaning to carry. The field of " + \
        # "transport has several aspects, loosely they can be divided into a triad of infrastructure, vehicles, and operations. " + \
        # "Infrastructure includes the transport networks (roads, railways, airways, canals, pipelines, etc.) that are used, as well as " + \
        # "the nodes or terminals (such as airports, train stations, bus stations and seaports). The vehicles generally ride on the " + \
        # "networks, such as automobiles, trains, airplanes. The operations deal with the control of the system, such as traffic signals " + \
        # "and ramp meters, railroad switches, air traffic control, etc, as well as policies, such as how to finance the system " + \
        # "(e.g use of tolls or gasoline taxes in the case of highway transport). Broadly speaking, the design of networks are " + \
        # " the domain of civil engineering and urban planning, the design of vehicles of mechanical engineering and specialized " + \
        # "subfields such as nautical engineering and aerospace engineering, and the operations are usually specialized, though might" + \
        # "appropriately belong to operations research or systems engineering. [/INST]\n"

    # eval_prompt = """
    #     Summarize this dialog:
    #     A: Hi Tom, are you busy tomorrow’s afternoon?
    #     B: I’m pretty sure I am. What’s up?
    #     A: Can you go with me to the animal shelter?.
    #     B: What do you want to do?
    #     A: I want to get a puppy for my son.
    #     B: That will make him so happy.
    #     A: Yeah, we’ve discussed it many times. I think he’s ready now.
    #     B: That’s good. Raising a dog is a tough issue. Like having a baby ;-)
    #     A: I'll get him one of those little dogs.
    #     B: One that won't grow up too big;-)
    #     A: And eat too much;-))
    #     B: Do you know which one he would like?
    #     A: Oh, yes, I took him there last Monday. He showed me one that he really liked.
    #     B: I bet you had to drag him away.
    #     A: He wanted to take it home right away ;-).
    #     B: I wonder what he'll name it.
    #     A: He said he’d name it after his dead hamster – Lemmy  - he's  a great Motorhead fan :-)))
    #     ---
    #     Summary:
    #     """
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    # # model.eval()
    with torch.no_grad():
        print(tokenizer.decode(model.generate(**model_input, max_new_tokens=1024)[0], skip_special_tokens=True))

    # output_jsons = []
    # data = json.load(open(args.input))
    # for json_instance in data:
    #     input_ = json_instance["r_content"]
    #     eval_prompt = f"Simplify the following text:\n{input_}\n---\nSimplified text:\n"   
    #     model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    #     with torch.no_grad():
    #         # print(eval_prompt)
    #         generated_text = model.generate(**model_input,
    #                                         max_new_tokens=100,
    #                                         num_beams=5,
    #                                         early_stopping=True)[0]
    #                                         # do_sample=False,
    #                                         # top_p=1.0)[0]
    #         generated_text = tokenizer.decode(generated_text, skip_special_tokens=True)
    #         print(generated_text)
    #         # output = {
    #         #     "input": input_,
    #         #     "prompt": eval_prompt,
    #         #     "output": generated_text
    #         # }
    #         print("*******************\n")
    #         break


                        


                




