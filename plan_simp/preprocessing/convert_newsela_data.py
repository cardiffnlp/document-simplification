import json, re
import argparse


def remove_reading_level(input_string):
    pattern = r'\.en-\d+'
    return re.sub(pattern, '.en', input_string)


def get_doc_content(content):
    new_content = {}
    for key, value in content.items():
        new_content[remove_reading_level(key)] = value
    return new_content


if __name__ == '__main__':

    # prepare argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    data = json.load(open(args.input))
    final_json = {}
    levels = ["0", "1", "2", "3", "4"]
    for doc_id, doc_info in data.items():
        # print(doc_id)
        for ni in range(len(levels)):
            for nj in range(ni + 1, len(levels)):
                # print(i), print(j)
                i = levels[ni]
                j = levels[nj]
                normal_id = doc_id + "-" + i
                simple_id = doc_id + "-" + j

                new_alignments = []
                for normal, simple in doc_info['sentence_alignment']:
                    if normal.startswith(normal_id) and simple.startswith(simple_id):
                        # print(simple, normal)
                        simple = remove_reading_level(simple)
                        normal = remove_reading_level(normal)
                        new_alignments.append([simple, normal])
                        # print(simple, normal)

                new_instance = {
                    "normal": {
                        "id": normal_id,
                        "title": normal_id,
                        "content": get_doc_content(doc_info[i])
                    },
                     "simple": {
                        "id": simple_id,
                        "title": simple_id,
                        "content": get_doc_content(doc_info[j])
                    },
                    "sentence_alignment": new_alignments
                }

                pair_id = doc_id + "-" + i + "-" + j
                final_json[pair_id] = new_instance 

    with open(args.output, 'w') as f:
        json.dump(final_json, f)
    
