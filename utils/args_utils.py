import argparse
from typing import Union

class Config(object):
    seed: int = 114514
    epochs: int = 200
    batch_size: int = 1
    resume_path: Union[str,None] = None
    image_size: tuple 
    data_path: str 
    warmup: int = 5

def get_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="data path", default="D:/datasets/ISLES/dataset-ISLES22^public^unzipped^version")
    parser.add_argument("--resume_path", type=str, help="resume path", required=False)
    parser.add_argument("--seed", type=int, help="seed", default=114514)
    parser.add_argument("--epochs", type=int, help="epochs", default=200)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=1)
    parser.add_argument("--image_length", type=int, help="image_length", default=96)
    parser.add_argument("--image_width", type=int, help="image_width", default=96)
    parser.add_argument("--image_height", type=int, help="image_height", default=96)
    parser.add_argument("--wramup", type=int, help="wramup step", default=5)

    args = parser.parse_args()

    Config.seed = args.seed
    Config.epochs = args.epochs
    Config.batch_size = args.batch_size
    Config.resume_path = args.resume_path
    Config.image_size = (args.image_length,args.image_width,args.image_height)
    Config.data_path = args.data
    Config.warmup = args.wramup

    return Config
