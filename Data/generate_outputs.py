"""
    Script to generate standard output format for ConditionalQA evaluation.
"""
import re
import json
import argparse


def exact_match(es: set, cs: set):
    return len(es.intersection(cs)) # same denominator: len(cs)

def find_condition(docs: list, doc_sets: list, con: str):
    cs = set([c.strip() for c in re.sub(r"<[^<>]+>", "", con).strip().split(' ')])
    # compute the EM score for each tag
    em, best = -1, con
    for i, d in enumerate(doc_sets):
        score = exact_match(d, cs)
        if score > em:
            em = score
            best = docs[i]
    return best

def line_to_dict(
        l: str, 
        ans_infix: str=' [A] ', con_prefix: str=' [C] ', con_infix: str=' [C] '
    ):
    qid, ans = l.split('\t', 1)
    as_and_cs = ans.split(ans_infix)
    out_ans = list()
    # eliminate repetitive answers
    a_all = set()
    for a_and_c in as_and_cs:
        if not a_and_c:
            continue
        a_and_cs = a_and_c.split(con_prefix)
        if len(a_and_cs) > 1:
            a = a_and_cs[0].strip()
            cs = [c.strip() for c in a_and_cs[1].split(con_infix)]
        else:
            a = a_and_cs[0].strip()
            cs = []
        if a not in a_all:
            a_all.add(a)
            out_ans.append([a, cs])
        else:
            for u in out_ans:
                if u[0] == a:
                    u[1].extend(cs)
    return {'id': qid, 'answers': out_ans}


def main(args):
    doc_filepath = f"{args.dataset}/documents.json"
    qs_filepath = f"{args.dataset}/{args.subset}.json"
    # load documents
    docs = {u['url']: (
        u['contents'], 
        [set([t.strip() for t in re.sub(r"<[^<>]+>", "", e).strip().split(' ')]) 
        for e in u['contents']]
    ) for u in json.load(open(doc_filepath, 'r'))}
    # qid-url mapping
    qid2url = {u['id']: u['url'] for u in json.load(open(qs_filepath, 'r'))}
    # build output filename
    out_filepath = args.filepath.replace('.txt', '_cqa.json')

    # load separators according to output format
    con_prefix, con_infix = ' [CON] ', ' [CON] '
    ans_infix = ' [SEP] '
    if args.build_natural:
        con_prefix, con_infix = ' IF ', ' AND '
        ans_infix = ' OR '

    # build output dictionary 
    output = list()
    lines = [l.strip() for l in open(args.filepath, 'r')]
    for l in lines:
        base = line_to_dict(l.strip(), ans_infix, con_prefix, con_infix)
        doc, doc_set = docs[qid2url[base['id']]]
        # match context
        ans = list()
        if base['answers']:
            for a_and_cs in base['answers']:
                a, cs = a_and_cs
                # remove tags if remained
                a = re.sub(r"<[^<>]+>", "", a).strip()
                cs = set([find_condition(doc, doc_set, c) for c in cs])
                ans.append([a, list(cs)])
        base['answers'] = ans
        output.append(base)
        
    # write out the result
    json.dump(output, open(out_filepath, 'w'), indent=4)
    print(f"\nResults successfully written to [{out_filepath}].\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converting output formats.')
    parser.add_argument(
        '--dataset',
        '-d',
        type=str,
        default='ConditionalQA/v1_0',
        help='Filepath to the original dataset folder.'
    )
    parser.add_argument(
        '--subset',
        '-s',
        type=str,
        choices=['dev', 'train', 'test_no_answer'],
        default='dev',
        help='Which subset of the data is used.'
    )
    parser.add_argument(
        '--filepath',
        '-f',
        type=str,
        help='Filepath to the generated outputs from FiD.'
    )
    parser.add_argument(
        '--build-natural',
        '-n',
        action='store_true',
        help='Whether the outputs are trained with natural language connections.'
    )
    args = parser.parse_args()
    main(args)

