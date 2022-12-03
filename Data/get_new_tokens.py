"""
    get all html tags from the training corpus
"""
import re
import json
import argparse
from tqdm import tqdm


def main(args):
    # load all the documents
    docs = json.load(open(f"{args.root}/documents.json", 'r'))
    # concate all the contents
    elements = [e for psg in docs for e in psg['contents']]
    tags = set()
    batch_bar = tqdm(total=len(elements), leave=False, dynamic_ncols=True, desc='reading...')
    # iterate
    for e in elements:
        results = re.findall(r"<[^<>]+>", e)
        tags = tags.union(set(results))
        batch_bar.set_postfix(total_tags=len(tags))
        batch_bar.update()
    # save all to the output directory
    with open(args.out, 'w') as fw:
        fw.write('\n'.join(list(tags)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finding all tags in documents.')
    parser.add_argument(
        '--root',
        '-r',
        type=str,
        default='./ConditionalQA/v1_0',
        help='Root directory of the ConditionalQA dataset.'
    )
    parser.add_argument(
        '--out',
        '-o',
        type=str,
        default='./Data/tags.txt',
        help='Output directory to save all the tags.'
    )
    args = parser.parse_args()
    main(args)