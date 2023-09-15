import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_path", type=str, help="resume path",required=False)

    return parser.parse_args()
