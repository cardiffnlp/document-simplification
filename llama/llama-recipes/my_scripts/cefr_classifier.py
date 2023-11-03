import re
import argparse
import json
import numpy as np
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from collections import Counter 
from string import punctuation


def modify_token(org_token, tag, vocab_rules, lemmatizer):

    if tag.startswith("J"):
        token = lemmatizer.lemmatize(org_token.lower(), pos ="a")
    elif tag.startswith("R"):
        token = lemmatizer.lemmatize(org_token.lower(), pos ="r")
    elif tag.startswith("V"):
        token = lemmatizer.lemmatize(org_token.lower(), pos ="v")
    else:
        token = lemmatizer.lemmatize(org_token.lower())
        if token == org_token:
            token = lemmatizer.lemmatize(org_token.lower(), pos ="a")
        if token == org_token:
            token = lemmatizer.lemmatize(org_token.lower(), pos ="v")
        if token == org_token:
            token = lemmatizer.lemmatize(org_token.lower(), pos ="r")


    if token not in vocab_rules and "the " + token in vocab_rules:
        token = "the " + token

    return token

def least_label(labels):
    if "A1" in labels:
        return "A1"
    elif "A2" in labels:
        return "A2"
    elif "B1" in labels:
        return "B1"
    elif "B2" in labels:
        return "B2"
    elif "C1" in labels:
        return "C1"
    else:
        return "C2"


def map_pos(input_pos):
    if input_pos.startswith("JJ"):
        return "adjective"
    elif input_pos.startswith("VB"):
        return "verb"
    elif input_pos.startswith("MD"):
        return "modal verb"
    elif input_pos.startswith("RB"):
        return "adverb"
    elif input_pos.startswith("PRP$"):
        return "determiner"
    elif input_pos.startswith("WP") or input_pos.startswith("PRP") or input_pos.startswith("WRB"):
        return "pronoun"
    elif input_pos.startswith("NN"):
        return "noun"
    elif input_pos.startswith("IN"):
        return "preposition"
    elif input_pos.startswith("DT"):
        return "determiner"
    else:
        return ""


def get_cefr_labels(text, vocab_rules, lemmatizer):

    text = text.split("\n")
    text = " ".join([para for para in text if len(para) > 0])

    cefr_labels = []
    cefr_word_labels = []
    text = word_tokenize(text.strip().replace("-", " "))
    tagged_text = pos_tag(text)
    for org_token, tag in tagged_text:
        
        token = modify_token(org_token, tag, vocab_rules, lemmatizer)
        
        if token in vocab_rules:
            mapped_pos = map_pos(tag)
            if mapped_pos in vocab_rules[token]:
                label = least_label(vocab_rules[token][mapped_pos])
            elif "" in vocab_rules[token]:
                label = least_label(vocab_rules[token][""])
            else:
                label = least_label([least_label(value) for value in vocab_rules[token].values()])
            
            cefr_labels.append(label)
            cefr_word_labels.append((org_token, token, label, tag))

        else:
            if token not in punctuation and token.isalnum():
                label = "none"
                cefr_labels.append(label)
                cefr_word_labels.append((org_token, token, label, tag))

    counter = Counter(cefr_labels)
    for key in counter:
        counter[key] = counter[key] * 100.0 / len(cefr_labels)
    return counter, cefr_word_labels


def get_cefr_rules(html_file):
    html_content = open(html_file).read().strip()
    html_content = html_content.replace(' style="white-space:normal"', "")

    vocab_rules = {}
    for content in re.findall("<tr>(.*?)</tr>", html_content, re.DOTALL):
        rows = [td for td in re.findall("<td>(.*?)</td>", content, re.DOTALL)]
        if len(rows) > 0:
            phrase = rows[0].strip().lower()
            topic = rows[1].strip()
            pos = rows[3].strip().replace("auxiliary", "").strip()
            label = rows[2].replace("</span>", "")[-2:].strip()
            vocab_rules.setdefault(phrase, {})
            vocab_rules[phrase].setdefault(pos, set())
            vocab_rules[phrase][pos].add(label)

    return vocab_rules


def get_cefr_levels(vocab_rules, text):

    lemmatizer = WordNetLemmatizer()
    cefr_labels, _ = get_cefr_labels(text, vocab_rules, lemmatizer)
    return cefr_labels
