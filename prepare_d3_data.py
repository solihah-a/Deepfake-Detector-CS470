import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from project_config import dataset_presets


def export_split(split_name, split_data, output_root, image_format, max_per_class):
    split_out = output_root / split_name
    real_dir = split_out / "real"
    fake_dir = split_out / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    per_class_count = {0: 0, 1: 0}
    total = split_data.num_rows

    for idx in tqdm(range(total), desc=f"exporting {split_name}"):
        row = split_data[idx]
        label = int(row["Label_A"])

        if label not in (0, 1):
            continue

        if max_per_class is not None and per_class_count[label] >= max_per_class:
            continue

        image = row["Image"]
        target_dir = real_dir if label == 0 else fake_dir
        target_path = target_dir / f"{split_name}_{idx:07d}.{image_format}"
        image.save(target_path)
        per_class_count[label] += 1

        if max_per_class is not None and all(
            per_class_count[c] >= max_per_class for c in (0, 1)
        ):
            break

    print(
        f"{split_name}: saved real={per_class_count[0]}, fake={per_class_count[1]} -> {split_out}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="export a deepfake dataset into d3 real/fake image folders."
    )
    parser.add_argument(
        "--dataset",
        default="dataset_one",
        choices=sorted(dataset_presets.keys()),
        help="pick a saved dataset setup.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="folder with train, validation, and test parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="output folder with split/real and split/fake folders.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="optional max number of real and fake images per split.",
    )
    parser.add_argument(
        "--image-format",
        default="jpg",
        choices=["jpg", "png"],
        help="saved image format.",
    )
    args = parser.parse_args()

    preset = dataset_presets[args.dataset]
    data_dir_value = args.data_dir or str(preset["source_data_dir"])
    output_dir_value = args.output_dir or str(preset["prepared_data_dir"])
    max_per_class = args.max_per_class
    if max_per_class is None:
        max_per_class = preset["max_per_class"]

    data_dir = Path(data_dir_value)
    output_root = Path(output_dir_value)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"dataset: {args.dataset}")
    print(f"data dir: {data_dir}")
    print(f"output dir: {output_root}")
    print(f"max per class: {max_per_class}")

    data_files = {
        "train": str(data_dir / "train-*.parquet"),
        "validation": str(data_dir / "validation-*.parquet"),
        "test": str(data_dir / "test-*.parquet"),
    }

    ds = load_dataset("parquet", data_files=data_files)

    for split_name in ["train", "validation", "test"]:
        export_split(
            split_name,
            ds[split_name],
            output_root,
            image_format=args.image_format,
            max_per_class=max_per_class,
        )


if __name__ == "__main__":
    main()
