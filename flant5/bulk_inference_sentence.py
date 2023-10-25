import json
import nltk
import torch
import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument('--instruct_flag', action='store_true')
    parser.add_argument('--coedit_flag', action='store_true')
    parser.add_argument('--zero_shot_flag', action='store_true')
    args = parser.parse_args()

    input_file = args.input 
    coedit_flag = args.coedit_flag
    model = args.model
    instruct_flag = args.instruct_flag
    zero_shot_flag = args.zero_shot_flag
    output_file = args.output

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()


    index = 0
    output_jsons = []
    data = json.load(open(input_file))
    for json_instance in data:
        input_paragraph = json_instance["r_content"]

        if len(input_paragraph) > 10000:
            output = {
                    "input": input_paragraph,
                    "prompt": "",
                    "output": "",
                    "target": json_instance["s_content"],
                }
            output_jsons.append(output)
            index += 1
            continue

        if coedit_flag:
            main_instruction = "Make the text grammatical, rewrite with different wording, and make this text less complex: "
        else:
            main_instruction = "Make the following text simpler: "

        input_sentences = nltk.sent_tokenize(input_paragraph)
        output_sentences = []
        for input_ in input_sentences:

            if instruct_flag and not zero_shot_flag:
                eval_prompt = main_instruction + \
                    "A T-shirt (or tee shirt) is a shirt, usually buttonless, collarless, and pocketless, with a round neck and short sleeves, that is pulled on over the head and covers most of a person's torso.\n" + \
                    "Simplified text:\n" + \
                    "A T-shirt or tee shirt is a kind of shirt which has short sleeves. These sleeves cover the shoulders and the top of the arm, but they do not cover the elbow or the forearm.\n\n" + \
                    main_instruction + input_ + "\nSimplified text:\n"
            else:
                eval_prompt = main_instruction + input_

            model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                generated_text = model.generate(**model_input,
                                                max_new_tokens=1024,
                                                do_sample=True,
                                                # repetition_penalty=1.1, top_p=0.9
                                                )[0]
                generated_text = tokenizer.decode(generated_text, skip_special_tokens=True)
                generated_text = generated_text.replace("<SEP>", "").replace("SEP>", "")
                output_sentences.append(generated_text)

        output_sentences = " ".join(output_sentences)
        output = {
                "input": input_paragraph,
                "output": " ".join(output_sentences),
                "target": json_instance["s_content"],
        }
        output_jsons.append(output)

        if index % 20 == 0:
            print(index, input_sentences)
            print(eval_prompt)
            print(output_sentences)
            print("********")
        index += 1
        # if index == 5:
        #     break

    with open(output_file, "w") as fp:
        json.dump(output_jsons, fp)
