from pathlib import Path


project_root = Path(__file__).resolve().parent
default_dataset_name = "dataset_one"
default_transfer_run_name = "full_data_transfer"


dataset_presets = {
    "dataset_one": {
        "source_data_dir": project_root / "data" / "Defactify_Image_Dataset" / "data",
        "prepared_data_dir": project_root / "data" / "d3_defactify",
        "max_per_class": None,
        "description": "full prepared dataset",
    },
    "dataset_two": {
        "source_data_dir": project_root / "data" / "Defactify_Image_Dataset" / "data",
        "prepared_data_dir": project_root / "data" / "d3_defactify_sample",
        "max_per_class": 5,
        "description": "small prepared sample dataset",
    },
}


model_presets = {
    "d3_pretrained": {
        "checkpoint": project_root / "D3" / "ckpt" / "classifier.pth",
        "clip_name": "ViT-L/14",
        "shuffle_times": 1,
        "original_times": 1,
        "patch_size": 14,
    },
}
