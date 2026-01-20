import argparse
from pathlib import Path

from src.preprocess import create_preprocess
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--override", type=Path, nargs="*", default=None, help="额外配置文件，后加载覆盖前者")
    args, cli_overrides = parser.parse_known_args()

    cfg = load_config(args.config, args.override, cli_overrides)
    preprocessor = create_preprocess(cfg)
    preprocessor.run()


if __name__ == "__main__":
    main()
