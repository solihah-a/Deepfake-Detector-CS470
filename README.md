# Deepfake Project

This project helps you do four main things:

1. turn the Defactify parquet files into image folders
2. run the pretrained D3 model
3. make a simple accuracy chart
4. run a small transfer learning baseline

## Before You Start

This repo already has these default settings:

- `dataset_one` = full prepared dataset at `data/d3_defactify`
- `dataset_two` = small sample dataset at `data/d3_defactify_sample`
- pretrained D3 checkpoint = `D3/ckpt/classifier.pth`

You can change those defaults in `project_config.py`.

## Setup

Open PowerShell in this folder:

```powershell
cd c:\Users\ngwaamj\Desktop\deepfake
.venv\Scripts\Activate.ps1
```

## Quick Start

### Step 1: Prepare the Data

For the full dataset:

```powershell
python prepare_d3_data.py --dataset dataset_one
```

For a small quick test:

```powershell
python prepare_d3_data.py --dataset dataset_two
```

This creates folders like this:

- `train/real`
- `train/fake`
- `validation/real`
- `validation/fake`
- `test/real`
- `test/fake`

If you want your own input or output folder:

```powershell
python prepare_d3_data.py --data-dir data/Defactify_Image_Dataset/data --output-dir data/my_dataset
```

### Step 2: Run the Pretrained D3 Model

Use the full prepared test set:

```powershell
cd D3
python validate_for_robustness.py
cd ..
```

Use the sample dataset instead:

```powershell
cd D3
python validate_for_robustness.py --test_folders ../data/d3_defactify_sample/test
cd ..
```

This saves results in `outputs/d3_pretrained_results`.

Important files:

- `outputs/d3_pretrained_results/validation.xlsx`
- `outputs/d3_pretrained_results/None_None_ypred.npz`
- `outputs/d3_pretrained_results/None_None_ytrue.npz`

### Step 3: Make a Simple Chart

```powershell
python analyze_results.py
```

This prints simple accuracy numbers and saves:

- `outputs/simple_accuracy_comparison.png`

If your result files are in a different folder:

```powershell
python analyze_results.py --results-dir outputs/my_results --plot-path outputs/my_plot.png --title "my results"
```

## Train D3 Yourself (Optional)

If you want to train D3 on your prepared dataset:

```powershell
cd D3
$env:D3_DATA_ROOT = "../data/d3_defactify"
python train.py --name defactify_run --checkpoints_dir ../checkpoints --arch CLIP:ViT-L/14 --fix_backbone --head_type attention --shuffle --patch_size 14
cd ..
```

This saves checkpoints in `checkpoints/defactify_run`.

To test that trained checkpoint:

```powershell
cd D3
python validate_for_robustness.py --ckpt ../checkpoints/defactify_run/model_epoch_best.pth
cd ..
```

## Transfer Learning Baseline (Optional)

`transfer_learning.py` is a very simple baseline.

It does this:

- loads the pretrained D3 attention head
- freezes the backbone
- trains only the attention head
- tests the model before training
- tests the model again after training

Run it like this:

```powershell
python transfer_learning.py
```

By default it uses:

- dataset: `data/d3_defactify`
- output folder: `outputs/transfer_learning/basic`

It saves:

- `outputs/transfer_learning/basic/results.csv`
- `outputs/transfer_learning/basic/best_attention_head.pth`

If you want different paths or training settings, edit the values near the top of `transfer_learning.py`.

## Compare Transfer Learning Results

After transfer learning finishes, run:

```powershell
python compare_transfer_results.py
```

This reads:

- `outputs/transfer_learning/basic/results.csv`

And saves:

- `outputs/transfer_learning/basic/before_after_bar_graph.png`

If you want a different input or output file, edit `RESULTS_PATH` and `PLOT_PATH` in `compare_transfer_results.py`.

## Colab

If you want to run this in Google Colab, install the packages first:

```python
!bash colab_setup.sh
```

or:

```python
!pip install -r requirements-colab.txt
```

Then check that Colab can see the GPU:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
```

## File Guide

- `prepare_d3_data.py`: turns parquet files into D3-style image folders
- `D3/train.py`: trains D3 on prepared data
- `D3/validate_for_robustness.py`: runs a checkpoint on a test folder and saves result files
- `analyze_results.py`: makes a simple accuracy chart from saved results
- `transfer_learning.py`: runs the simple transfer learning baseline
- `compare_transfer_results.py`: makes a before-vs-after transfer learning chart
- `project_config.py`: stores the default dataset paths and checkpoint path
