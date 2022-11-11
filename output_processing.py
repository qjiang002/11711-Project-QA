import json
import argparse

def parse_arguments():
    # command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--FiD_output_file', dest='pred_file', type=str,
                        default=None, help="Path to FiD output file.")
    parser.add_argument('--CQA_output_file', dest='ref_file', type=str,
                        default=None, help="Path to the converted output file for CQA evaluation.")
    return parser.parse_args()

def converter_without_conditions(FiD_output_file, CQA_output_file):
    json_outputs = []
    with open(FiD_output_file, 'r') as f:
        for line in f.readlines():
            id = line.split('\t')[0]
            ans = '\t'.join(line.strip().split('\t')[1:])
            if ans == 'not_answerable':
                out = {"id": id, "answers": []}
            else:
                out = {"id": id, "answers": [[ans, []]]}
            json_outputs.append(out)

    with open(CQA_output_file, 'w') as f:
        json.dump(json_outputs, f, indent=4)

if __name__=="__main__":
    args = parse_arguments()
    converter_without_conditions(args.FiD_output_file, args.CQA_output_file)