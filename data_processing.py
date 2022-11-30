import json
import copy 
import argparse


def generate_FiD_data_without_conditions(cqa_data_path, FiD_data_path, use_section_as_passage, question_as_passage):
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

  def content_to_sentences(contents):
    return [{"title": "", "text": sent} for sent in contents]

  # cqa_data_path = './ConditionalQA/v1_0/train.json'
  with open(cqa_data_path, 'r') as f:
    cqa_data = json.load(f)

  documents_path = './ConditionalQA/v1_0/documents.json'
  with open(documents_path, 'r') as f:
    documents = json.load(f)

  doc_dict = dict()
  for doc in documents:
    doc_dict[doc['url']] = {"title": doc['title'], "sections": content_to_sections(doc['contents']) if use_section_as_passage else content_to_sentences(doc['contents'])}

  FiD_examples = []
  for example in cqa_data:
    new_example = dict()
    new_example['id'] = example['id']
    new_example['question'] = example['question'] if not question_as_passage else ""
    if 'test' not in cqa_data_path:
      target = ' [SEP] '.join([ans[0] for ans in example['answers']]) if example['not_answerable'] == False else 'unanswerable'
      new_example['target'] = target
      new_example['answers'] = [target]
    else:
      new_example['target'] = ""
      new_example['answers'] = [""]
    new_example['ctxs'] = copy.deepcopy(doc_dict[example['url']]['sections'])
    if question_as_passage:
      new_example['ctxs'].append({'title': "", 'text': example['scenario']})
      new_example['ctxs'].append({'title': "", 'text': example['question']})
    else:
      # new_example['ctxs'].append({'title': "title", 'text': doc_dict[example['url']]['title']})
      new_example['ctxs'].append({'title': "scenario", 'text': example['scenario']})
    FiD_examples.append(new_example)

  # FiD_data_path = './FiD_train.json'
  with open(FiD_data_path, 'w') as f:
    json.dump(FiD_examples, f, indent=4)


def clean_html(sent):
  sent = sent.strip().split('>')[1]
  clean_sent = sent.split('</')[0]
  return clean_sent

def generate_FiD_data(cqa_data_path, FiD_data_path, include_title=True, conj_symbol='[SEP]', remove_html=True):
  def content_segmentation(title, contents, include_title, remove_html):
    tokens = []
    for sent in contents:
      clean_sent = clean_html(sent) if remove_html else sent
      tokens.extend(clean_sent.split(' '))
    
    ans = []
    seg = []
    seg_length = 100
    for i, t in enumerate(tokens):
      seg.append(t)
      if i==len(tokens)-1 or (i+1)%seg_length==0:
        ans.append({"title": (title if include_title else ""), "text": ' '.join(seg)})
        seg = []
    return ans

  def answers_to_target_SEP(answers, remove_html):
    if len(answers)==0:
      return "unanswerable"
    segs = []
    for ans in answers:
      segs.append(ans[0])
      if len(ans[1])>0:
        segs.extend([clean_html(sent) if remove_html else sent for sent in ans[1]])
      else:
        segs.append('NA')
    return ' [SEP] '.join(segs)

  def answer_to_target_CON(answers, remove_html):
    if len(answers)==0:
      return "unanswerable"
    segs = []
    for ans in answers:
      target_str = ans[0] + ' [CON] '
      if len(ans[1])>0:
        target_str += ' [CON] '.join([clean_html(sent) if remove_html else sent for sent in ans[1]])
      else:
        target_str += 'NA'
      segs.append(target_str)
    return ' [SEP] '.join(segs)


  with open(cqa_data_path, 'r') as f:
    cqa_data = json.load(f)

  documents_path = './ConditionalQA/v1_0/documents.json'
  with open(documents_path, 'r') as f:
    documents = json.load(f)
  
  doc_dict = dict()
  for doc in documents:
    doc_dict[doc['url']] = {"title": doc['title'], "sections": content_segmentation(doc['title'], doc['contents'], include_title, remove_html)}

  FiD_examples = []
  for example in cqa_data:
    new_example = dict()
    new_example['id'] = example['id']
    new_example['question'] = example['scenario'] + ' [SEP] ' + example['question']
    if 'test' not in cqa_data_path:
      new_example['target'] = answers_to_target_SEP(example['answers'], remove_html) if conj_symbol=='[SEP]' else answer_to_target_CON(example['answers'], remove_html)
      new_example['answers'] = [new_example['target']]
    else:
      new_example['target'] = ""
      new_example['answers'] = [""]
    new_example['ctxs'] = copy.deepcopy(doc_dict[example['url']]['sections'])
    FiD_examples.append(new_example)

  with open(FiD_data_path, 'w') as f:
    json.dump(FiD_examples, f, indent=4)
  

def parse_arguments():
    # command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--without_condition', action='store_true', help="Only convert answers.")
    parser.add_argument('--no_title', action='store_true', help="FiD contexts don't have titles.")
    parser.add_argument('--use_section_as_passage', action='store_true', help="For without_condition, use each sentence as a passage in FiD.")
    parser.add_argument('--question_as_passage', action='store_true', help="For without_condition, treat question and scenario as passages.")
    parser.add_argument('--conj_symbol', type=str, default='[SEP]', help='symbol that seperates answer and conditions, [SEP] or [CON].')
    parser.add_argument('--keep_html', action='store_true', help="Don't remove the HTML headers.")
    parser.add_argument('--cqa_data_path', type=str, default='./ConditionalQA/v1_0/train.json', help='Path to original ConditionalQA json data.')
    parser.add_argument('--FiD_data_path', type=str, default='./FiD_train_SEP_title.json', help='Path to FiD input data.')
    return parser.parse_args()

if __name__=="__main__":
  args = parse_arguments()
  if args.without_condition:
    generate_FiD_data_without_conditions(args.cqa_data_path, args.FiD_data_path,  args.use_section_as_passage, args.question_as_passage)
  else:
    generate_FiD_data(args.cqa_data_path, args.FiD_data_path, include_title=(not args.no_title), conj_symbol=args.conj_symbol, remove_html=(not args.keep_html))
