"""
    Driver script to generate training data in the FiD format for ConditionalQA.
        choice: evidences only / sections of different levels / whole passage
"""
import re
import json
import argparse


class dynamicSplitter:
    total_count = 0
    super_long_count = 0

    def __init__(self, token_limit: int=300, error: int=20):
        self.token_limit = token_limit
        # allowing 5 tokens for finer tokenization
        self.error = error
        # compute the max length for each "passage" to input in FiD
        self.max_len = self.token_limit - self.error

    def split_elements(self, q_len: int, elements: list):
        """
            Splitting the input HTML elements to fit into T5 length limit.
        """
        outs, split = list(), False
        cur_strs, cur_len  = list(), q_len
        for e in elements:
            e_len = len(e.split(' '))
            if e_len + cur_len < self.max_len:
                cur_len += e_len
                cur_strs.append(e)
            else:
                outs.append(' '.join(cur_strs))
                split = True
                cur_strs, cur_len = [e], e_len + q_len
        self.total_count += 1
        self.super_long_count += int(split)
        # append the last string into the output
        outs.append(' '.join(cur_strs))
        return outs


def main(args):
    # obtain the filepaths to training and development sets
    train_filepath = f"{args.root}/train.json"
    dev_filepath = f"{args.root}/dev.json"

    # load the documents if the entire sections are needed
    docs = (json.load(open(f"{args.root}/documents.json", 'r')) 
            if (args.use_segments or args.whole_doc) else None)
    if docs:
        # convert to dictionary for faster indexing
        docs = {u['url']: u for u in docs}
    
    # build dynamic splitter
    splitter = dynamicSplitter(args.token_limit)

    def build_one_set(
            in_filepath: str, out_filepath: str,
            use_segments: bool=False, segment_level: int=3,
            add_conditions: bool=True, build_natural: bool=True,
            keep_all: bool=True, cond_only: bool=True,
            whole_doc: bool=False
        ):
        # specification of concatenation tags
        con_prefix, con_infix = ' [C] ', ' [C] '
        sce_prefix, q_prefix = '[S] ', ' [Q] '
        ans_infix = ' [A] '
        if build_natural:
            con_prefix, con_infix = ' IF ', ' AND '
            sce_prefix, q_prefix = 'GIVEN THAT ', ' THEN '
            ans_infix = ' OR '
        # load original data
        original = json.load(open(in_filepath, 'r'))
        # define output variable
        out = list()
        for qdict in original:
            if cond_only and (
                (len(qdict['answers']) == 1 and not qdict['answers'][0][1]) 
                or qdict['not_answerable']
            ):
                continue
            answers = ([f"{u[0]}{con_prefix + con_infix.join(u[1]) if u[1] else ''}"
                       if add_conditions else u[0] if qdict['answers'] else '' for u in qdict['answers']]
                       if not qdict['not_answerable'] else ['[N/A]']) 
            # concatenate all answers to be 1 single target
            answers = ans_infix.join(answers) if len(answers) > 1 else answers[0]
            if not build_natural:
                answers = ans_infix[1:] + answers
            unit = {'id': qdict['id'],
                    # concatenate scenario with question
                    'question': f"{sce_prefix}{qdict['scenario']}{q_prefix}{qdict['question']}",
                    'target': answers, 'answers': [answers]}
            # compute rough question token length
            q_len = len(unit['question'].split(' '))
            if whole_doc:
                contexts = splitter.split_elements(q_len=q_len,
                                                   elements=docs[qdict['url']]['contents'])
                unit['ctxs'] = [{'title': '', 'text': ctx} for ctx in contexts]
            elif not use_segments:
                if qdict['evidences']:
                    evidences = splitter.split_elements(q_len=q_len,
                                                        elements=qdict['evidences'])
                else:
                    evidences = ['']
                unit['ctxs'] = [{'title': '', 'text': e} for e in evidences]
            else:
                contents = docs[qdict['url']]['contents']
                evidences = set(qdict['evidences'])
                # obtain the levels of contents
                lvs = {c[:4] for c in contents if c.startswith('<h')}
                segment_lv = len(lvs) if segment_level > len(lvs) else segment_level
                candidates = set()
                tag_to_segment = dict()
                unit_segment = list()
                last_tag = ''
                for c in contents:
                    if c.startswith('<h') and int(c[2]) <= segment_lv:
                        if last_tag:
                            tag_to_segment[last_tag] = unit_segment
                        last_tag = c
                        unit_segment = list()
                    if c in evidences and last_tag:
                        candidates.add(last_tag)
                    if last_tag and c != last_tag:
                        unit_segment.append(c)
                # add the last tag if exists
                if last_tag:
                    tag_to_segment[last_tag] = unit_segment
                if not candidates:
                    assert not evidences
                # build contexts
                unit['ctxs'] = list()
                for tag in candidates:
                    t_len = len(tag.split(' '))
                    split_contexts = splitter.split_elements(q_len=q_len + t_len, 
                                                             elements=tag_to_segment[tag])
                    unit['ctxs'].extend([{'title': tag, 'text': sc} for sc in split_contexts])
                # add irrelevant segments as "unknown" disturbance
                if (not cond_only) and keep_all:
                    extra = {k: v for k, v in unit.items()}
                    extra['target'] = '[N/A]'
                    extra['answers'] = ['[N/A]']
                    rest = set(tag_to_segment.keys()) - candidates
                    for tag in rest:
                        new = {k: v for k, v in extra.items()}
                        # add tag to total token length count as well
                        t_len = len(tag.split(' '))
                        split_contexts = splitter.split_elements(q_len=q_len + t_len, 
                                                                 elements=tag_to_segment[tag])
                        new['ctxs'] = [{'title': tag, 'text': sc} for sc in split_contexts]
                        if new['ctxs']:
                            out.append(new)
            # add question unit
            if unit['ctxs']:
                out.append(unit)

        json.dump(out, open(out_filepath, 'w'), indent=4)
        # show statistics
        print(f"In [{out_filepath}], there are [{splitter.super_long_count}/{splitter.total_count}] "
               "evidences split for being too long.")

    # output directory
    out_dir = f"{args.out}/with{'' if args.add_conditions else 'out'}_conditions"
    tag = ((f"segments_lv{args.segment_level}" if args.use_segments 
            else 'evidences' if not args.whole_doc else 'full')
           + ('_natural' if args.build_natural else '')
           + ('_keepall' if args.keep_all else '')
           + ('_onlycon' if args.cond_only else ''))
    
    # only doing so for training and development set since test set has no "evidence" tab
    for subset in ('train', 'dev'):
        in_filepath = f"{args.root}/{subset}.json"
        out_filepath = f"{out_dir}/fineReading_{subset}_{tag}.json"
        build_one_set(
            in_filepath=in_filepath, out_filepath=out_filepath, 
            use_segments=args.use_segments, segment_level=args.segment_level,
            add_conditions=args.add_conditions, keep_all=args.keep_all,
            build_natural=args.build_natural, cond_only=args.cond_only,
            whole_doc=args.whole_doc
        )



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generating training data for FiD.')
    parser.add_argument(
        '--root',
        '-r',
        type=str,
        default='./ConditionalQA/v1_0',
        help='Root directory of the ConditionalQA dataset.'
    )
    parser.add_argument(
        '--build-natural',
        '-n',
        action='store_true',
        help='Whether to concatenate answers and conditions in natural language.'
    )
    parser.add_argument(
        '--keep-all',
        '-a',
        action='store_true',
        help='Whether to keep all segments in the passage with irrelavant ones marked as unknown.'
    )
    parser.add_argument(
        '--cond-only',
        '-j',
        action='store_true',
        help='Whether to keep only questions with conditional answers.'
    )
    parser.add_argument(
        '--use-segments',
        '-s',
        action='store_true',
        help='Whether to use the segment (not the evidence) as the retrieved inputs.'
    )
    parser.add_argument(
        '--add-conditions',
        '-c',
        action='store_true',
        help='Whether to add conditions (with segmenting tags) to the answers.'
    )
    parser.add_argument(
        '--segment-level',
        '-l',
        type=int,
        default=3,
        help='Level of granularity to extract the segment.'
    )
    parser.add_argument(
        '--whole-doc',
        '-w',
        action='store_true',
        help='Whether to use the entire passage.'
    )
    parser.add_argument(
        '--token-limit',
        '-t',
        type=int,
        default=300,
        help='Total limit of token lengths due to T5 capacity.'
    )
    parser.add_argument(
        '--out',
        '-o',
        type=str,
        default='./Data',
        help='Output directory of generated training data for FiD.'
    )

    args = parser.parse_args()
    main(args)
    