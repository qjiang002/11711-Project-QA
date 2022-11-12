import json
import argparse
from evaluate import load
exact_match_metric = load("exact_match")

documents_path = './ConditionalQA/v1_0/documents.json'
with open(documents_path, 'r') as f:
    documents = json.load(f)

doc_dict = dict()
for doc in documents:
    doc_dict[doc['url']] = {"title": doc['title'], "contents": doc['contents']}

def match_condition(contents, cond):
    max_score, ans = 0, ""
    for sent in contents:
        score = exact_match_metric.compute(predictions=[cond], references=[sent], ignore_punctuation=True)['exact_match']
        if score > max_score:
            max_score = score
            ans = sent
    return ans

def parse_answer(contents, answer, conj_symbol='[SEP]'):
    answer = [a.strip() for a in answer.strip().split('[SEP]')]
    if answer[0].startswith('unanswerable'):
        return []
    # contents = doc_dict[example_id_url[example_id]]["contents"]
    if conj_symbol=='[SEP]':
        res = []
        segs = []
        for token in answer:
            if token == 'NA':
                if len(segs)==1:
                    res.append([segs[0], []])
                else:
                    print(segs)
                    res.append([segs[0], [match_condition(contents, cond) for cond in segs[1:]]])
                segs = []
            else:
                segs.append(token)
        if len(segs)>0:
            if len(segs)==1:
                res.append([segs[0], []])
            else:
                res.append([segs[0], [match_condition(contents, cond) for cond in segs[1:]]])
        return res
    res = []
    for ans in answer:
        print(ans)
        ans = [a.strip() for a in ans.split('[CON]')]
        if len(ans)<=1:
            res.append([ans[0], []])
        else:
            conds = [match_condition(contents, c) for c in ans[1:] if c!='NA']
            res.append(ans[0], conds)
    return res

def convert(cqa_data_path, FiD_output_file, CQA_output_file, without_conditions=False):
    # cqa_data_path = './ConditionalQA/v1_0/dev.json'
    with open(cqa_data_path, 'r') as f:
        cqa_data = json.load(f)

    example_id_url = dict()
    for example in cqa_data:
        example_id_url[example['id']] = example['url']

    json_outputs = []
    with open(FiD_output_file, 'r') as f:
        for line in f.readlines():
            id = line.split('\t')[0]
            ans = '\t'.join(line.strip().split('\t')[1:])
            if without_conditions:
                if ans == 'not_answerable':
                    out = {"id": id, "answers": []}
                else:
                    out = {"id": id, "answers": [[ans, []]]}
            else:
                contents = doc_dict[example_id_url[id]]["contents"]
                out = {"id": id, "answers": parse_answer(contents, ans)}
            json_outputs.append(out)

    with open(CQA_output_file, 'w') as f:
        json.dump(json_outputs, f, indent=4)


def parse_arguments():
    # command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--FiD_output_file', type=str,
                        default=None, help="Path to FiD output file.")
    parser.add_argument('--CQA_output_file', type=str,
                        default=None, help="Path to the converted output file for CQA evaluation.")
    parser.add_argument('--cqa_data_path', type=str,
                        default=None, help="Path to original ConditionalQA data file.")
    parser.add_argument('--without_conditions', action='store_true', help="Trained with data in Data/without_conditions.")
    return parser.parse_args()

if __name__=="__main__":
    args = parse_arguments()
    convert(args.cqa_data_path, args.FiD_output_file, args.CQA_output_file, args.without_conditions)