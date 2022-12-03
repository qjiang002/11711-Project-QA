"""
    Script to generate standard output format for ConditionalQA evaluation.
"""

import json
import argparse


def fix_partial(c: str):
    if not c.startswith('<'):
        c = '<' + c
    if c.endswith('>'):
        i = len(c) - 1
        while c[i] != '/':
            i -= 1
        if c[i - 1] != '<':
            c = c[:i] + '<' + c[i:]
    return c


def line_to_dict(
        l: str, ans_infix: str=' [A] ',
        con_prefix: str=' [C] ', con_infix: str=' [C] '
    ):
    qid, ans = l.split('\t', 1)
    as_and_cs = ans.split(ans_infix[1:])
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
            # manually add left bracket if not present
            cs = [fix_partial(c) for c in cs]
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
    # build output filename
    out_filepath = args.filepath.replace('.txt', '_cqa.json')

    # load separators according to output format
    con_prefix, con_infix = ' [C] ', ' [C] '
    sce_prefix, q_prefix = '[S] ', ' [Q] '
    ans_infix = ' [A] '
    if args.build_natural:
        con_prefix, con_infix = ' IF ', ' AND '
        sce_prefix, q_prefix = 'GIVEN THAT ', ' THEN '
        ans_infix = ' OR '

    # write out the result
    json.dump(
        [line_to_dict(l.strip()) for l in open(args.filepath, 'r')], 
        open(out_filepath, 'w'), indent=4
    )
    print(f"\nResults successfully written to [{out_filepath}].\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converting output formats.')
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

