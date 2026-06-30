# Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation

This repository contains the implementation of the ICLR 2026 paper **Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation**.

## Model

![model](img/model.png)

## Requirements

```bash
conda create -n ADAlign python=3.10 pip
conda activate ADAlign

pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 scikit-learn==1.6.1 pandas cvxpy
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install torch-geometric==2.6.1 pygda==1.2.0
```

Use the matching PyTorch/PyG wheels if your CUDA version is different.

## Datasets

Download datasets from the links in [data/README.md](data/README.md), then place them as:

```text
data/
  Citation/ACMv9/
  Citation/Citationv1/
  Citation/DBLPv7/
  Airport/USA/
  Airport/BRAZIL/
  Airport/EUROPE/
  Blog/Blog1/
  Blog/Blog2/
  Twitch/DE/
  Twitch/EN/
```

## Training

Run one transfer:

```bash
python main.py --source ACMv9 --target Citationv1 --device cuda:0 --runs 5
```

Short CPU smoke test:

```bash
python main.py --synthetic --device cpu --runs 1 --epochs 1 --t_batchsize 32 --verbose 0
```

Paper aliases are supported:

```text
A=ACMv9, C=Citationv1, D=DBLPv7
U=USA, B=BRAZIL, E=EUROPE
B1=Blog1, B2=Blog2
```

## Main Arguments

| Argument | Meaning |
| --- | --- |
| `--source`, `--target` | Source and target domains |
| `--device` | `cuda:0`, `cuda:1`, or `cpu` |
| `--runs` | Number of repeated runs |
| `--lr` | Learning rate |
| `--weight_decay` | Weight decay |
| `--t_batchsize` | Frequency number |
| `--s_pnums`, `--t_pnums` | Propagation steps |
| `--weight` | Alignment loss weight |
| `--dropout` | Dropout ratio |
| `--nhid` | Hidden feature dimension |
| `--alpha` | Amplitude weight |
| `--epochs` | Training epochs |

Reference best configurations selected from search runs are provided in [best_hyperparameters.json](best_hyperparameters.json).

## Search Space

The Table 7 search space is implemented in [configs.py](configs.py).

| Hyperparameter | Values |
| --- | --- |
| learning rate | `[0.001, 0.005, 0.01]` |
| weight decay | `[0.001, 0.005, 0.01]` |
| frequency number | `[128, 1024, 2048, 4096]` |
| pnums | `[0, 1, 10, 15]` |
| lambda | `[0, 0.2, 0.4, 0.6, 0.8, 1]` |
| dropout ratio | `[0.1, 0.25, 0.5]` |
| feature dimension | `128` |
| amplitude weight | `[0, 0.2, 0.4, 0.6, 0.8, 1]` |
| epochs | `150` |

Some transfers use the expanded candidate sets from Table 7, such as additional frequency numbers for Blog/Twitch and additional Airport candidates. See `configs.py` for the exact per-transfer space.

## Hyperparameter Search

Random search:

```bash
python scripts/grid_search.py \
  --source A \
  --target C \
  --device cuda:0 \
  --strategy random \
  --max-trials 100 \
  --runs 1 \
  --output-dir results/search/A_to_C_random
```

Full grid search:

```bash
python scripts/grid_search.py \
  --source A \
  --target C \
  --device cuda:0 \
  --strategy grid \
  --runs 1 \
  --output-dir results/search/A_to_C_grid
```

Search outputs include:

```text
trial_*.json
trials.csv
best_config.json
```

The default selection metric is `micro_f1_mean`. Use `--metric macro_f1_mean` to select by Macro-F1.

## Multi-GPU Search

For all paper transfer tasks:

```bash
python scripts/launch_paper_search.py \
  --gpus 0,1,2,3 \
  --strategy random \
  --max-trials 1000 \
  --runs 1 \
  --output-root results/paper_search
```

Summarize and validate best configurations:

```bash
python scripts/summarize_search.py results/paper_search
python scripts/validate_all_best.py results/paper_search --runs 5 --gpus 0,1,2,3
```

## Citation

```bibtex
@inproceedings{adalign2026,
  title = {Learning Adaptive Distribution Alignment with Neural Characteristic Function for Graph Domain Adaptation},
  booktitle = {International Conference on Learning Representations},
  year = {2026}
}
```
