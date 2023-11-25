import json, argparse, glob
import simplediff
from nltk import word_tokenize

from colorama import Fore, Back
from termcolor import colored, cprint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    args = parser.parse_args()

    print_flag = False
    for fpath in glob.glob(args.input):
        print(fpath)
        data = json.load(open(fpath))
        instance = data[120]
            
        if True:
        # for instance in data:
            input = instance['input']
            output = instance['output']

            print(instance['input'])
            print()
            print(Fore.BLACK, output)
            print()
            print(Fore.BLACK, instance['target'][0])
            print()

            output = output.split("\n")
            output = " ".join([para for para in output if len(para) > 0])
            edits = simplediff.diff(word_tokenize(input), word_tokenize(output))

            final_text = []
            for op, tokens in edits:
                if op == "=":
                    print(Fore.BLACK + " ".join(tokens) + " ", end="")
                elif op == "-":
                    print(Fore.RED + " ".join(tokens) + " ", end="")
                else:
                    print(Fore.GREEN + " ".join(tokens) + " ", end="")
            print(" ".join(final_text))
            print(Fore.BLACK, "\n")
