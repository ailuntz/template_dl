import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_raw_data(raw_dir: Path, train_size: int, val_size: int,
                      test_size: int, num_features: int, num_classes: int) -> None:
    """生成原始CSV数据"""
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("生成原始CSV数据...")
    np.random.seed(42)

    for split, size in [('train', train_size), ('val', val_size), ('test', test_size)]:
        data = np.random.randn(size, num_features)
        labels = np.random.randint(0, num_classes, size)

        df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(num_features)])
        df['label'] = labels

        csv_path = raw_dir / f'{split}.csv'
        df.to_csv(csv_path, index=False)
        print(f"  - {csv_path}: {data.shape}")


def preprocess(raw_dir: Path, out_dir: Path, train_size: int, val_size: int,
               test_size: int, num_features: int, num_classes: int) -> None:
    """处理原始CSV数据并保存到processed目录"""
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_raw_data(raw_dir, train_size, val_size, test_size, num_features, num_classes)

    print("\n处理CSV数据 (归一化)...")

    train_df = pd.read_csv(raw_dir / 'train.csv')
    val_df = pd.read_csv(raw_dir / 'val.csv')
    test_df = pd.read_csv(raw_dir / 'test.csv')

    feature_cols = [col for col in train_df.columns if col != 'label']

    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std()

    for df, split in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        df[feature_cols] = (df[feature_cols] - mean) / (std + 1e-8)

        out_path = out_dir / f'{split}.csv'
        df.to_csv(out_path, index=False)
        print(f"  - {out_path}: {df.shape}")

    print(f"\n完成! 数据已保存到:")
    print(f"  - raw: {raw_dir}")
    print(f"  - processed: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train_size", type=int, default=256)
    parser.add_argument("--val_size", type=int, default=64)
    parser.add_argument("--test_size", type=int, default=64)
    parser.add_argument("--num_features", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()
    preprocess(args.raw_dir, args.out_dir, args.train_size, args.val_size,
               args.test_size, args.num_features, args.num_classes)


if __name__ == "__main__":
    main()
