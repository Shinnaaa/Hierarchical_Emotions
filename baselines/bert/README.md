# Baseline: BERT fine-tuned with BCE

The fine-tuned baseline of the paper: BERT (`bert-base-cased`) for 28-label GoEmotions classification, trained with binary cross-entropy only. It is [monologg/GoEmotions-pytorch](https://github.com/monologg/GoEmotions-pytorch) with one addition: evaluation also reports the **EMD** metric with the hierarchical cost matrix (`cost_matric.py`), so the baseline is scored exactly like the proposed method in the repository root.

This folder was merged from the former `Shinnaaa/GoEmotions-baseline` fork with its full history; the upstream code and its Apache-2.0 [license](LICENSE) are kept.

## Run

Run from this folder. The config reads the shared dataset in `../../data/original`.

```bash
cd baselines/bert
pip install -r requirements.txt
python run_goemotions.py --taxonomy original
```

Checkpoints go to `ckpt/original/`; evaluation prints accuracy, micro/macro/weighted F1, Hamming loss and EMD.

## Differences from upstream

| File | Change |
|---|---|
| `cost_matric.py` | New: emotion hierarchy and the label-to-label cost matrix |
| `utils.py` | `compute_metrics` adds Hamming loss and EMD over sigmoid outputs |
| `run_goemotions.py` | Collects sigmoid outputs during evaluation and passes them to the metrics |
| `config/original.json` | `data_dir` points to the shared `../../data/original` |

Hyperparameters (in `config/original.json`) match upstream: learning rate 5e-5, batch size 16, 10 epochs, max sequence length 50, warmup 0.1, threshold 0.3.
