import json
import copy 

def generate_FiD_data_without_conditions(cqa_data_path, FiD_data_path):
  def content_to_sections(contents):
    ans = []
    sec = []
    for sent in contents:
      if sent.startswith('<h'):
        if len(sec)>0:
          ans.append({"title": sec[0], "text": (' '.join(sec[1:]) if len(sec)>1 else "")})
        sec = []
      sec.append(sent)
    if len(sec)>0:
      ans.append({"title": sec[0], "text": (' '.join(sec[1:]) if len(sec)>1 else "")})
    return ans

  # cqa_data_path = './ConditionalQA/v1_0/train.json'
  with open(cqa_data_path, 'r') as f:
    cqa_data = json.load(f)

  documents_path = './ConditionalQA/v1_0/documents.json'
  with open(documents_path, 'r') as f:
    documents = json.load(f)

  doc_dict = dict()
  for doc in documents:
    doc_dict[doc['url']] = {"title": doc['title'], "sections": content_to_sections(doc['contents'])}

  FiD_examples = []
  for example in cqa_data:
    new_example = dict()
    new_example['id'] = example['id']
    new_example['question'] = example['question']
    if 'test' not in cqa_data_path:
      new_example['answers'] = [ans[0] for ans in example['answers']] if example['not_answerable'] == False else ['not_answerable']
    new_example['ctxs'] = copy.deepcopy(doc_dict[example['url']]['sections'])
    new_example['ctxs'].append({'title': "title", 'text': doc_dict[example['url']]['title']})
    new_example['ctxs'].append({'title': "scenario", 'text': example['scenario']})
    FiD_examples.append(new_example)

  # FiD_data_path = './FiD_train.json'
  with open(FiD_data_path, 'w') as f:
    json.dump(FiD_examples, f, indent=4)


if __name__=="__main__":
  cqa_data_path = './ConditionalQA/v1_0/train.json'
  FiD_data_path = './FiD_train.json'
  generate_FiD_data_without_conditions(cqa_data_path, FiD_data_path)
