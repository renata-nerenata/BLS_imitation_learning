import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Inference for puzzle prediction')
    parser.add_argument('--puzzle', '--p')
    parser.add_argument('--model', '-m',
                        help='Model for prediction',
                        default='transformer')
    args = parser.parse_args()
    return args
