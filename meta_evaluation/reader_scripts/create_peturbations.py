import re, csv
import json
import glob
import random
import argparse

from fuzzywuzzy import fuzz
from nltk import sent_tokenize, word_tokenize
from transformers import AutoTokenizer, T5ForConditionalGeneration

# TOKENIZER = AutoTokenizer.from_pretrained("grammarly/coedit-xl-composite")
# MODEL = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-xl-composite").to("cuda:0")

# TOKENIZER = AutoTokenizer.from_pretrained("grammarly/coedit-xl-composite")
# MODEL = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-xl-composite").to("cuda:0")

def peturb(text, type="coherence"):
    sentences = sent_tokenize(text)
    if type == "coherence":
        N = len(sentences)
        indices = random.sample([i for i in range(N)], max(int(0.3 * N), 2))
        new_sentences = []
        for index, sentence in enumerate(sentences):
            if index not in indices:
                new_sentences.append(sentence)
        random.shuffle(indices)
        new_sentences.extend([sentences[ind] for ind in indices])
        return " ".join(new_sentences) 
    elif type == "deletion":
        half_num = int(len(sentences) * 0.8)
        return " ".join(sentences[:half_num])
    elif type == "fluency":
        N = len(sentences)
        indices = random.sample([i for i in range(N)], int(0.5 * N))
        assert len(indices) > 0
        new_sentences = [] 
        for index, sentence in enumerate(sentences):
            tokens = word_tokenize(sentence)
            if index in indices and len(tokens) > 7:
                index = random.randint(0, len(tokens) - 6)
                inter = tokens[index:index+6]
                random.shuffle(inter)
                tokens = tokens[:index] + inter + tokens[index+6:]
                new_sentence = " ".join(tokens).replace(" .", ".").replace(" ,", ",")
                new_sentences.append(new_sentence)
            else:
                new_sentences.append(sentence)   
        return " ".join(new_sentences)
    elif type == "punctuation":
        N = len(sentences)
        indices = random.sample([i for i in range(N)], int(0.5 * N))
        new_sentences = []
        for index, sentence in enumerate(sentences):
            sentence = sentence.replace(",", "")
            if index in indices:
                tokens = word_tokenize(sentence)
                num_tokens = int(len(tokens) * 0.5)
                tokens = tokens[:num_tokens] + ["."] + tokens[num_tokens:]
                sentence = " ".join(tokens).replace(" .", ".").replace(" ,", ",")
            new_sentences.append(sentence)
        return " ".join(new_sentences)
    elif type == "paraphrasing":
        N = len(sentences)
        indices = random.sample([i for i in range(N)], max(int(0.2 * N), 2))
        new_sentences = []
        for index, sentence in enumerate(sentences):
            if index in indices:
                input_text = "Paraphrase this: " + sentences[index]
                input_ids = TOKENIZER(input_text, return_tensors="pt").input_ids.to("cuda:0")
                outputs = MODEL.generate(input_ids, max_length=256)
                sentence = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
            new_sentences.append(sentence)
        return " ".join(new_sentences) 
    elif type == "repetition":
        N = len(sentences)
        indices = random.sample([i for i in range(N)], max(int(0.2 * N), 2))
        new_sentences = []
        for index, sentence in enumerate(sentences):
            if index in indices:
                if index % 2 == 0:
                    input_text = "Paraphrase this: " + sentences[index]
                    input_ids = TOKENIZER(input_text, return_tensors="pt").input_ids.to("cuda:0")
                    outputs = MODEL.generate(input_ids, max_length=256)
                    new_sentence = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
                    sentence = sentence + " " + new_sentence
                else:
                    tokens = word_tokenize(sentence)
                    index = random.randint(0, len(tokens) - 4)
                    inter = tokens[index:index+4]
                    tokens = tokens[:index] + inter + inter + tokens[index+6:]
                    sentence = " ".join(tokens)
            new_sentences.append(sentence)
        return " ".join(new_sentences)


def peturb_indomain(simplification, reference, dataset):
    sentences = sent_tokenize(reference.lower())
    for text in dataset:
        text_sentences = sent_tokenize(text)
        if any(sent in text.lower() for sent in sentences):
            end_index = -1
            for index, text_sent in enumerate(text_sentences):
                if fuzz.ratio(text_sent.lower(), sentences[-1]) > 95.0:
                    end_index = index
                    break
            next_sent = text_sentences[end_index + 1] + " " + text_sentences[end_index + 2]
            return simplification + " " + next_sent


def peturb_outdomain(simplification, dataset):
    index = random.randint(0, len(dataset) - 1)
    text_sentences = sent_tokenize(dataset[index])
    sent_index = random.randint(0, len(text_sentences) - 1)
    return simplification + " " + text_sentences[sent_index]
   
                        
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--human")
    parser.add_argument("--peturb")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    args=parser.parse_args()

    with open(args.human) as fp:
        csvreader = csv.reader(fp)
        headers = next(csvreader)
        dataset = {}
        for row in csvreader:
            instance = {h:r for h, r in zip(headers, row)}
            key = instance['article_id'], instance['para_id']
            dataset.setdefault(key, {})
            system = instance['Model']
            dataset[key][system] = instance

        with open(args.output, 'w') as fp:
            for _, instance in dataset.items():
                original = instance['Original']['paragraph']
                references = [instance['Elementary']['paragraph']]
                simplification = instance['ChatGPT']['paragraph']

                if args.peturb == "copy":
                    peturbed_simplification = peturb(original, "paraphrasing")
                elif args.peturb == "indomain" or args.peturb == "outdomain":
                    dataset = []
                    for fpath in glob.glob(args.dataset + "/*"):
                        doc_text = open(fpath).read().strip()
                        doc_text = " ".join(doc_text.split("\n"))
                        dataset.append(doc_text)
                    if args.peturb == "indomain":
                        peturbed_simplification = peturb_indomain(simplification, references[0], dataset)
                    else:
                        peturbed_simplification = peturb_outdomain(simplification, dataset)
                    assert peturbed_simplification is not None
                elif args.peturb == "outdomain":
                    peturb_outdomain(simplification, dataset)
                else:
                    peturbed_simplification = peturb(simplification, args.peturb)
                
                data_json = {}
                data_json["original"] = original
                data_json["rating_type"] = "pairwise"
                data_json["references"] = references
                data_json["simplification1"] = simplification
                data_json["simplification2"] = peturbed_simplification
                data_json["system1"] = "ChatGPT"
                data_json["system2"] = "ChatGPT-peturbed"
                ratings = {
                    "name": args.peturb,
                    "type": "binary",
                    "value": [0],
                    "agg_value": 0
                }
                data_json["ratings"] = [ratings]
                fp.write(json.dumps(data_json) + "\n")

                print(original)
                print(simplification)
                print(peturbed_simplification)
                print()