import json
from scipy.special import softmax
import copy 
import argparse

def content_to_sections(contents):
    ans = []
    sec = []
    for sent in contents:
      if sent.startswith('<h'):
        if len(sec)>0:
          ans.append({"title": "", "text": (' '.join(sec))})
        sec = []
      sec.append(sent)
    if len(sec)>0:
      ans.append({"title": "", "text": (' '.join(sec))})
    return ans

def content_to_sentences(contents):
    return [{"title": "", "text": sent} for sent in contents]

def generate_condition_FiD_data(level, dataset_score_file, cqa_data_path, FiD_data_path, use_score):
    if use_score:
        with open(dataset_score_file, 'r') as f:
            passage_scores = json.load(f)
        score_dict = {}
        for item in passage_scores:
            scores = [p['score'] for p in item['ctxs']]
            probs = softmax(scores)
            min_prob, max_prob = min(probs), max(probs)
            score_dict[item['id']] = {p['text'].strip(): (max_prob - probs[i])/(max_prob - min_prob) for i, p in enumerate(item['ctxs'])}
    
    with open(cqa_data_path, 'r') as f:
        cqa_data = json.load(f)
    
    documents_path = './ConditionalQA/v1_0/documents.json'
    with open(documents_path, 'r') as f:
        documents = json.load(f)
    doc_dict = dict()
    for doc in documents:
        doc_dict[doc['url']] = {"title": doc['title'], "sections": content_to_sections(doc['contents']) if level=='section' else content_to_sentences(doc['contents'])}

    FiD_examples = []
    for example in cqa_data:
        for i, ans in enumerate(example['answers']): # skip unanswerable examples
            new_example = dict()
            new_example['id'] = example['id'] + '_' + str(i)
            new_example['question'] = example['question'] + ' [SEP] ' + ans[0]
            if 'test' not in cqa_data_path:
                target = ' [SEP] '.join(ans[1]) if len(ans[1]) > 0 else 'NA'
                new_example['target'] = target
                new_example['answers'] = [target]
            else:
                new_example['target'] = ""
                new_example['answers'] = [""]
            new_example['ctxs'] = copy.deepcopy(doc_dict[example['url']]['sections'])
            if use_score:
                if level == 'section':
                    for ctx in new_example['ctxs']:
                        for k, v in score_dict[example['id']].items():
                            if k in ctx['text']:
                                ctx['title'] = str(v)[2]
                                break
                else:
                    for ctx in new_example['ctxs']:
                        ctx['title'] = str(score_dict[example['id']][ctx['text']])[2]
            new_example['ctxs'].append({'title': "scenario", 'text': example['scenario']})
            FiD_examples.append(new_example)
    
    with open(FiD_data_path, 'w') as f:
        json.dump(FiD_examples, f, indent=4)
    
def parse_arguments():
    # command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_score', action='store_true', help="Use cross attention scores.")
    parser.add_argument('--level', type=str, default='section', help='section / sentence.')
    parser.add_argument('--dataset_score_file', type=str, default='./dataset_wscores_section_level_train.json', help="Path to answer file with cross attention scores.")
    parser.add_argument('--cqa_data_path', type=str, default='./ConditionalQA/v1_0/train.json', help='Path to original ConditionalQA json data.')
    parser.add_argument('--FiD_data_path', type=str, default='./FiD_train_SEP_title.json', help='Path to FiD input data.')
    return parser.parse_args()

if __name__=="__main__":
  args = parse_arguments()
  generate_condition_FiD_data(args.level, args.dataset_score_file, args.cqa_data_path, args.FiD_data_path, args.use_score)

