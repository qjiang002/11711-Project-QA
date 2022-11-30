"""
    Driver script to generate training data in the FiD format for ConditionalQA.
        choice: evidences only
"""
import json
import argparse



def main(args):
    # obtain the filepaths to training and development sets
    train_filepath = f"{args.root}/train.json"
    dev_filepath = f"{args.root}/dev.json"

    # load the documents if the entire sections are needed
    docs = (json.load(open(f"{args.root}/documents.json", 'r')) 
            if args.use_segments else None)
    if docs:
        # convert to dictionary for faster indexing
        docs = {u['url']: u for u in docs}

    def build_one_set(
            in_filepath: str, out_filepath: str,
            use_segments: bool=False, segment_level: int=3,
            add_conditions: bool=True
        ):
        # load original data
        original = json.load(open(in_filepath, 'r'))
        # define output variable
        out = list()
        for qdict in original:
            answers = [f"{u[0]}{' [C] ' + ' [C] '.join(u[1]) if u[1] else ''}"
                       if add_conditions else u[0] if qdict['answers'] else '' for u in qdict['answers']]    
            unit = {'id': qdict['id'],
                    # concatenate scenario with question
                    'question': "[S] {} [Q] {}".format(qdict['scenario'], qdict['question']),
                    'answers': answers}
            if not use_segments:
                unit['ctxs'] = [{'title': '', 'text': e} if qdict['evidences'] else {'title': '', 'text': ''} 
                                for e in qdict['evidences']]
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
                        last_tag = ''
                        unit_segment = list()
                    if c.startswith(f"<h{segment_lv}>"):
                        last_tag = c
                    if c in evidences and last_tag:
                        candidates.add(last_tag)
                    if last_tag and c != last_tag:
                        unit_segment.append(c)
                # add the last tag if exists
                if last_tag:
                    tag_to_segment[last_tag] = unit_segment
                # build contexts
                unit['ctxs'] = [{'title': tag, 'text': ' '.join(tag_to_segment[tag])} for tag in candidates]
            # add question unit
            out.append(unit)
        json.dump(out, open(out_filepath, 'w'), indent=4)
        return out

    # output directory
    out_dir = f"{args.out}/with{'' if args.add_conditions else 'out'}_conditions"
    tag = f"segments_lv{args.segment_level}" if args.use_segments else 'evidences'

    # only doing so for training and development set since test set has no "evidence" tab
    for subset in ('train', 'dev'):
        in_filepath = f"{args.root}/{subset}.json"
        out_filepath = f"{out_dir}/fineReading_{subset}_{tag}.json"
        build_one_set(
            in_filepath=in_filepath, out_filepath=out_filepath, 
            use_segments=args.use_segments, segment_level=args.segment_level,
            add_conditions=args.add_conditions
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
        '--out',
        '-o',
        type=str,
        default='./Data',
        help='Output directory of generated training data for FiD.'
    )

    args = parser.parse_args()
    main(args)
    