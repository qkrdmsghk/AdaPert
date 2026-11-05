<div align="center">

# AdaPert

### Learning Adaptive Perturbation-Conditioned Contexts<br>for Robust Transcriptional Response Prediction

Yinhua Piao<sup>1</sup>, Hyomin Kim<sup>1</sup>, Seonghwan Kim<sup>1</sup>, Yunhak Oh<sup>1</sup>, Junhyeok Jeon<sup>2</sup>,<br>
Sang-Yeon Hwang<sup>2</sup>, Jaechang Lim<sup>2</sup>, Woo Youn Kim<sup>1,2</sup>, Chanyoung Park<sup>1</sup>, Sungsoo Ahn<sup>1†</sup>

<sup>1</sup>KAIST &emsp; <sup>2</sup>HITS &emsp; <sup>†</sup>Corresponding author

[![ICML 2026](https://img.shields.io/badge/ICML-2026-4b44ce.svg)](https://proceedings.mlr.press/v306/piao26a.html)
[![arXiv](https://img.shields.io/badge/arXiv-2602.18885-b31b1b.svg)](https://arxiv.org/abs/2602.18885)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-K562-2ea44f.svg)](../../releases)
[![License](https://img.shields.io/badge/License-Non--commercial-lightgrey.svg)](LICENSE)

[Method](#method) · [Installation](#installation) · [Data](#data) · [Evaluate](#evaluate-the-released-checkpoints) · [Train](#train-your-own-model) · [Citation](#citation)

</div>

<br>

This repository contains the code to train and evaluate AdaPert on the K562 Perturb-seq screen (Replogle et al., 2022), where the test perturbations are unseen during training.

> [!NOTE]
> This release covers **K562**: both the training config and the released checkpoints are for this cell line. Interested in other cell lines or your own data? Feel free to [open an issue](../../issues).

## Method

AdaPert predicts the response to a perturbation unseen in training while avoiding *mean collapse* (predicting a similar average response for every perturbation). It is adaptive and perturbation-conditioned in two ways:

1. **Adaptive context.** For each perturbation, a context selector scores the training perturbations near the perturbed gene in the STRING graph, conditioned on the perturbation, and keeps the relevant ones. The context is the weighted mean of their measured expression changes.
2. **Adaptive learning.** During training, the DEGs of each perturbation ([Data](#data), step 4) guide the loss: it suppresses spurious variation in non-responsive genes and emphasizes the DEGs.

Both are built on an encoder–decoder backbone: a basal-state encoder for the control cell, a perturbation encoder (a GNN over the STRING gene graph and GenePT text embeddings), and a decoder that combines them with the context.

<details>
<summary><b>Folder structure</b></summary>

```
.
├── main.py                        # training entry point (Hydra)
├── configs/adapert-k562.yaml      # K562 training configuration
├── calculate_DEGs.py              # DEGs used in training (Data, step 4)
├── scripts/
│   ├── get_predictions.py         # test-set predictions from a checkpoint
│   └── run_cell_eval.py           # cell-eval metrics
├── src/
│   ├── models/adapert.py          # AdaPert
│   ├── models/txpert.py           # encoder / GNN / decoder backbone
│   ├── predictor.py               # Lightning module (training and evaluation loop)
│   ├── metrics.py                 # validation and test metrics logged during training
│   └── data/                      # data module and knowledge graph
├── checkpoints/                   # released checkpoints (download, see Evaluate the released checkpoints)
└── research/                      # data (download, see Data)
    ├── data/
    │   ├── master_gene_set_sorted.csv
    │   ├── gears_gene_set.csv
    │   └── graphs/string/v11.5.parquet
    ├── cache/
    │   ├── K562_single_cell_line/
    │   │   ├── de_adata_test.h5ad
    │   │   └── splits/predefined/
    │   │       └── train_test_split.pkl
    │   └── GenePT_emebdding_v2/
    │       └── GenePT_gene_protein_embedding_model_3_text.pickle
    ├── AAA_DEGs/K562_single_cell_line/
    │   └── true_degs_dict.pkl
    └── checkpoints/K562_single_cell_line/   # checkpoints of your training runs
```

</details>

## Installation

```bash
conda create -n adapert python=3.12
conda activate adapert
# Install PyTorch and PyTorch Geometric (with torch_scatter) for your CUDA version first:
#   https://pytorch.org/get-started/locally/
#   https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.4.0 (CUDA 12.4), PyTorch Geometric 2.7.0, Lightning 2.5.1, pandas 2.2.3, scanpy 1.11.4, anndata 0.12.1, and cell-eval 0.5.43.

## Data

| Folder | Source |
|---|---|
| `research/data/` | `data/` folder of [TxPert](https://github.com/valence-labs/TxPert) |
| `research/cache/K562_single_cell_line/` | [Zenodo 15420279](https://zenodo.org/records/15420279) |
| `research/cache/GenePT_emebdding_v2/` | [Zenodo 10833191](https://zenodo.org/records/10833191) |
| `research/AAA_DEGs/` | `calculate_DEGs.py` (step 4) |

Run everything from the repository root. **Steps 1–3** are enough to evaluate the released checkpoints; **step 4** is only needed for training.

```bash
# 1. Knowledge graph and gene sets
git clone https://github.com/valence-labs/TxPert.git
mkdir -p research && cp -r TxPert/data research/data

# 2. K562 expression data and split
wget -O K562_single_cell_line.zip "https://zenodo.org/records/15420279/files/K562_single_cell_line.zip?download=1"
mkdir -p research/cache
unzip K562_single_cell_line.zip -d research/cache/
mkdir -p research/cache/K562_single_cell_line/splits/predefined
mv research/cache/K562_single_cell_line/splits/*.pkl research/cache/K562_single_cell_line/splits/predefined/

# 3. Gene embeddings (the archive stores this file with a trailing dot; restore the expected name)
wget -O GenePT_emebdding_v2.zip "https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1"
unzip GenePT_emebdding_v2.zip -d research/cache/
mv "research/cache/GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle." \
   research/cache/GenePT_emebdding_v2/GenePT_gene_protein_embedding_model_3_text.pickle

# 4. DEGs: Wilcoxon rank-sum test (scanpy) against the non-targeting cells, Benjamini-Hochberg adjusted p < 0.05.
#    About 2-3 hours on a CPU; re-running it skips the perturbations that are done.
cd research
python ../calculate_DEGs.py --dataset_name K562_single_cell_line
cd ..
```

<details>
<summary><b>Dataset statistics</b></summary>

| | Train | Validation | Test |
|---|---|---|---|
| Perturbations | 733 | 82 | 272 |
| Perturbed cells | 101,079 | 10,918 | 38,475 |
| DEGs per perturbation (median / mean) | 235 / 490 | 194 / 426 | 193 / 445 |

- **Control cells:** 10,691 non-targeting cells, part of the training split.
- **Genes:** 5,000 highly variable genes. `de_adata_test.h5ad` also contains 499 cells of 5 perturbations that are not in the split and are not used.
- **Knowledge graph:** STRING v11.5 over the 18,216 genes of `master_gene_set_sorted.csv`, keeping the 10 highest-confidence edges per gene: 171,250 directed edges. 17,133 genes have at least one edge, including 1,047 of the 1,087 perturbed genes.
- **Gene embeddings:** GenePT (3,072-d) covers all validation and test perturbations and 728 of the 733 training perturbations.

</details>

## Evaluate the released checkpoints

Download the three K562 checkpoints (training seeds 1–3) from [Releases](../../releases) into `checkpoints/`. Each checkpoint stores its own inference settings, so no config file is needed.

```bash
# 1. Predict the test perturbations
mkdir -p predictions
python scripts/get_predictions.py \
    --ckpt_path checkpoints/adapert-k562-seed2.ckpt \
    --output predictions/adapert_k562.h5ad

# 2. Compute cell-eval metrics
python scripts/run_cell_eval.py \
    --pred predictions/adapert_k562.h5ad \
    --real research/cache/K562_single_cell_line/de_adata_test.h5ad \
    --output eval_results/adapert_k562 \
    --name adapert_k562
```

The results are written to `eval_results/adapert_k562/`:

- `adapert_k562_summary.csv`: means over the 272 test perturbations
- `adapert_k562_results.csv`: per-perturbation results

<details>
<summary><b>SHA-256 checksums</b></summary>

| File | Training seed | SHA-256 |
|---|---|---|
| `adapert-k562-seed1.ckpt` | 1 | `2e0e6012fb20b35672a70789213286524e8312b93f273d9cc6eda538385341c9` |
| `adapert-k562-seed2.ckpt` | 2 | `530db317e91fb4c3df00163b4839eb12a1f11de221cd5655b59951c340e94dfa` |
| `adapert-k562-seed3.ckpt` | 3 | `1008439b5b3ec85e7fcbf09cfe6e0db924bac617492b7e848d575bebd470e79f` |

To check your downloads, compare the output of this command with the table:

```bash
sha256sum checkpoints/adapert-k562-seed*.ckpt
```

</details>

## Train your own model

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config-name adapert-k562
```

> [!NOTE]
> `configs/adapert-k562.yaml` is a reference setting for training your own model. The released checkpoints were trained with a different setting, so a model trained with this config won't match them. To get the checkpoints' numbers, evaluate them directly as shown [above](#evaluate-the-released-checkpoints).

- **Overrides:** any setting can be changed from the command line with Hydra, e.g. `seed=2`. No W&B account? Prefix the command with `WANDB_MODE=offline`.
- **Checkpoints:** validation runs every 2 epochs, and the best checkpoint is saved to `research/checkpoints/K562_single_cell_line/<wandb.name>_<run id>/`. To evaluate it, pass it as `--ckpt_path` to the commands [above](#evaluate-the-released-checkpoints).
- **Run time:** training stops after 25 validation checks without improvement, or after 200 epochs. This takes about 9 hours and 4 GB of memory on one GPU.

## Citation

If you find AdaPert useful in your research, please cite:

```bibtex
@inproceedings{piao2026adapert,
  title     = {Learning Adaptive Perturbation-Conditioned Contexts for Robust Transcriptional Response Prediction},
  author    = {Piao, Yinhua and Kim, Hyomin and Kim, Seonghwan and Oh, Yunhak and Jeon, Junhyeok and Hwang, Sang-Yeon and Lim, Jaechang and Kim, Woo Youn and Park, Chanyoung and Ahn, Sungsoo},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  pages     = {98696--98729},
  year      = {2026}
}
```

## Acknowledgements

This code is a modified version of [TxPert](https://github.com/valence-labs/TxPert) (commit `08d82ee`). We use the K562 Perturb-seq data of Replogle et al. (2022) with the processed data and split released by TxPert, the GenePT gene embeddings, the STRING database, and cell-eval for evaluation.

## License

For non-commercial research, academic and educational use only. This repository is a modified version of TxPert and, together with the released checkpoints, is distributed under the Recursion Pharmaceuticals Non-Commercial End User License Agreement ([TxPert-license.pdf](TxPert-license.pdf)). See [LICENSE](LICENSE) for details.

<details>
<summary>Attribution required by the license</summary>

> We used the TxPert AI model, available from Recursion Pharmaceuticals, with software documentation at www.rxrx.ai, pursuant to Recursion Pharmaceuticals' licensing terms at https://github.com/valence-labs/TxPert/blob/08d82eea86746b044cf7531f4ec8c5f60e1cb73f/license.pdf. Under this license, Recursion Pharmaceuticals disclaims all representations and warranties with respect to such AI model.

</details>
