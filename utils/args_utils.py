import argparse
from typing import Union, Optional, Tuple


class Config(object):
    def __init__(
        self,
        data_path: str,
        resume_path: Optional[str],
        image_size: Tuple[int],
        seed: int = 114514,
        epochs: int = 200,
        batch_siz: int = 1,
        warmup: int = 5,
    ) -> None:
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_siz
        self.resume_path = resume_path
        self.image_size = image_size
        self.data_path = data_path
        self.warmup = warmup


def get_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        help="data path",
        default="D:/datasets/ISLES/dataset-ISLES22^public^unzipped^version",
    )
    parser.add_argument("--resume_path", type=str, help="resume path", required=False)
    parser.add_argument("--seed", type=int, help="seed", default=114514)
    parser.add_argument("--epochs", type=int, help="epochs", default=200)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=1)
    parser.add_argument("--image_length", type=int, help="image_length", default=96)
    parser.add_argument("--image_width", type=int, help="image_width", default=96)
    parser.add_argument("--image_height", type=int, help="image_height", default=96)
    parser.add_argument("--wramup", type=int, help="wramup step", default=5)

    args = parser.parse_args()

    config = Config(
        data_path=args.data,
        resume_path=args.resume_path,
        image_size=(args.image_length, args.image_width, args.image_height),
        seed=args.seed,
        epochs=args.epochs,
        batch_siz=args.batch_size,
        warmup=args.wramup,
    )

    return config
