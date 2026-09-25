<p align="center">
  <a href="README.md"><img alt="English" src="https://img.shields.io/badge/English-1f2328?style=for-the-badge"></a>
  <a href="docs/README.zh-CN.md"><img alt="简体中文" src="https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-eaeef2?style=for-the-badge"></a>
  <a href="docs/README.ja.md"><img alt="日本語" src="https://img.shields.io/badge/%E6%97%A5%E6%9C%AC%E8%AA%9E-eaeef2?style=for-the-badge"></a>
</p>

# Emotional Earth Mover’s Distance for Fine-Grained Hierarchical Emotion Analysis

<p>
  <a href="https://link.springer.com/chapter/10.1007/978-981-95-3453-1_20"><img alt="Paper: ADMA 2025" src="https://img.shields.io/badge/paper-ADMA%202025-1f6feb"></a>
  <a href="https://doi.org/10.1007/978-981-95-3453-1_20"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.1007%2F978--981--95--3453--1__20-blue"></a>
  <img alt="Python 3.6+" src="https://img.shields.io/badge/python-3.6%2B-3776ab">
  <a href="LICENSE"><img alt="Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-green"></a>
</p>

Official code for **"Emotional Earth Mover’s Distance for Fine-Grained Hierarchical Emotion Analysis"**, Hai-Tao Yu, Dawei Li, Xin Kang. *Advanced Data Mining and Applications (ADMA 2025)*, Springer, pp. 296–310. [[paper]](https://link.springer.com/chapter/10.1007/978-981-95-3453-1_20)

Fine-grained emotion labels are not independent: predicting *amusement* for a *joy* comment is a smaller mistake than predicting *sadness*. Standard multi-label training (binary cross-entropy) treats both errors the same. This work proposes **Emotional Earth Mover's Distance (EEMD)**, which encodes the hierarchy of emotion labels as a transport cost, uses it as a training loss next to BCE, and uses it as an evaluation metric that rewards "near misses" over distant ones.

## What's in this repository

| Path | What it is | Role in the paper |
|---|---|---|
| repository root | BERT + learnable BCE/EEMD loss (this work) | Proposed method |
| [`baselines/bert/`](baselines/bert) | BERT with BCE only ([GoEmotions-pytorch](https://github.com/monologg/GoEmotions-pytorch)), extended with the EEMD evaluation metric | Fine-tuned baseline |
| [`baselines/llm/`](baselines/llm) | Chain-of-thought prompting of an LLM (gpt-3.5-turbo), scored with the same metrics | LLM baseline |
| [`data/original/`](data/original) | GoEmotions train/dev/test split with 28 labels, shared by all three | Dataset |

All three report micro/macro/weighted F1, accuracy, Hamming loss and EMD on the same split, so their numbers are directly comparable. The results are in the paper.

## Reproduce

```bash
pip install -r requirements.txt

# Proposed method (run from the repository root)
python run_goemotions.py --taxonomy original

# Fine-tuned BERT baseline (run from its folder; it reads ../../data/original)
cd baselines/bert && python run_goemotions.py --taxonomy original && cd ../..

# LLM baseline (evaluates the dev set; costs API credits)
export OPENAI_API_KEY=sk-...
python baselines/llm/Fine-grained-emotions-analysis-by-LLM.py
```

The BERT runs pin `torch==1.4.0` and `transformers==2.11.0`, as used for the paper; a Python 3.7 environment is the easiest way to install them. The LLM baseline needs `openai`, `pandas`, `POT` and `scikit-learn`, and accepts `OPENAI_MODEL` to try a different model.

## Overview

This repository implements a BERT-based multi-label emotion classification model that combines:
- **Softmax activation** for probability distribution over emotions
- **Earth Mover's Distance (EMD)** loss based on hierarchical emotion structure
- **Adaptive loss combination** using learnable weights to balance Binary Cross Entropy (BCE) and EMD losses

The core innovation is using EMD with a hierarchical cost matrix that encodes semantic distances between emotions, allowing the model to penalize predictions that are semantically distant from ground truth labels, not just incorrect.

## Key Features

- **Hierarchical Loss Function**: Combines traditional BCE loss with EMD loss based on emotion hierarchy
- **Multi-label Classification**: Handles multiple simultaneous emotion labels per input
- **BERT-based Architecture**: Built on top of BERT for robust text understanding
- **Learnable Loss Weighting**: Automatically learns the optimal balance between classification and hierarchical losses

## Installation

### Requirements

- Python 3.6+
- PyTorch 1.4.0
- Transformers 2.11.0
- Additional dependencies listed in `requirements.txt`

### Setup

```bash
# Clone the repository
git clone https://github.com/Shinnaaa/Hierarchical_Emotions.git
cd Hierarchical_Emotions

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `torch==1.4.0`
- `transformers==2.11.0`
- `attrdict==2.0.1`
- `numpy>=1.18.0`
- `scikit-learn>=0.22.0`
- `tqdm>=4.40.0`
- `POT>=0.7.0` (Python Optimal Transport library)
- `tensorboard>=2.0.0`

## Dataset

This project uses the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset, which contains:
- **58,000 Reddit comments** labeled with fine-grained emotions
- **28 emotion classes**: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, and neutral

### Dataset Structure

Place your dataset files in the `data/original/` directory:
- `train.tsv`: Training data (tab-separated: text \t label_ids \t comment_id)
- `dev.tsv`: Development/validation data
- `test.tsv`: Test data
- `labels.txt`: List of emotion labels (one per line)

## Model Architecture

### Hierarchical Emotion Structure

The emotions are organized in a hierarchical tree structure:
```
Root
├── positive
│   └── joy_lev2
│       ├── admiration
│       ├── amusement
│       ├── approval
│       └── ...
├── negative
│   ├── anger_lev2
│   ├── sadness_lev2
│   ├── disgust_lev2
│   └── fear_lev2
├── ambiguous
│   └── surprise_lev2
└── neutral
```

### Loss Function

The model uses a novel loss function that combines BCE and EMD:

```
L_final = L_BCE × (1 - α) + L_EMD × α
```

where:
- `L_BCE`: Binary Cross Entropy loss for multi-label classification
- `L_EMD`: Earth Mover's Distance loss based on hierarchical cost matrix
- `α`: Learnable weight (sigmoid output from a small neural network)

The EMD loss measures the minimum cost to transform the predicted emotion distribution into the ground truth distribution, where the cost is defined by the hierarchical distance between emotions.

## Usage

### Training

Train the model using the configuration file:

```bash
python run_goemotions.py --taxonomy original
```

The `--taxonomy` argument selects a configuration file from the `config/` directory. The paper uses `original` (28 fine-grained emotions), which is the configuration included here.

### Configuration

Hyperparameters can be adjusted in the configuration files located in `config/`:

```json
{
  "task": "goemotions",
  "data_dir": "data/original",
  "model_name_or_path": "bert-base-cased",
  "learning_rate": 5e-5,
  "num_train_epochs": 10,
  "train_batch_size": 16,
  "eval_batch_size": 32,
  "max_seq_len": 50,
  "warmup_proportion": 0.1,
  ...
}
```

### Evaluation

The model automatically evaluates on the development/test set during training. To evaluate a saved checkpoint:

```bash
# Set "do_eval": true and "do_train": false in config
python run_goemotions.py --taxonomy original
```

Evaluation metrics include:
- **Accuracy**: Exact match accuracy
- **Macro/Micro/Weighted F1**: Precision, recall, and F1 scores
- **Hamming Loss**: Average fraction of labels incorrectly predicted
- **EMD**: Earth Mover's Distance between predicted and true distributions

### Inference

For inference using the trained model, use the `MultiLabelPipeline` class:

```python
from transformers import BertTokenizer
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline

# Load model and tokenizer
model = BertForMultiLabelClassification.from_pretrained("path/to/checkpoint")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# Create pipeline
pipe = MultiLabelPipeline(model=model, tokenizer=tokenizer, threshold=0.3)

# Predict emotions
result = pipe("I'm so happy and excited about this!")
print(result)  # [{'labels': ['joy', 'excitement'], 'scores': [...]}]
```

## Project Structure

```
Hierarchical_Emotions/
├── baselines/
│   ├── bert/                  # BCE-only BERT baseline (see its README)
│   └── llm/                   # LLM chain-of-thought baseline (see its README)
├── config/
│   └── original.json          # Configuration files
├── data/
│   └── original/              # Dataset files
│       ├── train.tsv
│       ├── dev.tsv
│       ├── test.tsv
│       └── labels.txt
├── model.py                   # BERT model with hierarchical loss
├── data_loader.py             # Data loading and preprocessing
├── cost_matric.py             # Cost matrix computation for EMD
├── utils.py                   # Utility functions and metrics
├── multilabel_pipeline.py    # Inference pipeline
├── run_goemotions.py          # Main training/evaluation script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Key Components

### `model.py`
- `BertForMultiLabelClassification`: Main model class
  - Implements BERT encoder + classification head
  - Computes combined BCE and EMD loss
  - Uses learnable weight for loss combination

### `cost_matric.py`
- `compute_cost_matrix()`: Computes hierarchical cost matrix
- `hierarchy_distance()`: Calculates distance between emotions in hierarchy
- Defines the emotion hierarchy structure

### `run_goemotions.py`
- Training and evaluation loops
- Checkpoint management
- TensorBoard logging

### `utils.py`
- Evaluation metrics computation
- EMD computation for evaluation
- Logging and seed setting utilities

## Hyperparameters

Default hyperparameters (configurable in `config/*.json`):

| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| Learning Rate | 5e-5 | Initial learning rate for AdamW |
| Batch Size | 16 | Training batch size |
| Eval Batch Size | 32 | Evaluation batch size |
| Max Seq Length | 50 | Maximum sequence length |
| Epochs | 10 | Number of training epochs |
| Warmup Proportion | 0.1 | Fraction of training steps for warmup |
| Weight Decay | 0.0 | L2 regularization coefficient |

## Experimental Results

The full comparison of the proposed loss with the fine-tuned and LLM baselines, on all six metrics, is reported in the [paper](https://link.springer.com/chapter/10.1007/978-981-95-3453-1_20). Qualitatively, the hierarchical loss:
1. handles semantically related emotions better,
2. penalises predictions that are close in the hierarchy less than distant ones,
3. improves fine-grained distinctions within an emotion group.

## Technical Notes

### Why Softmax for Multi-label?
While softmax is typically used for single-label classification, we apply it here to create probability distributions that can be compared using EMD. The model still handles multiple labels through the one-hot label encoding and BCE loss.

### EMD with Hierarchical Cost Matrix
The cost matrix encodes semantic relationships: emotions sharing a common ancestor have lower costs. For example:
- `joy` and `excitement` (both under `joy_lev2`) have low cost
- `joy` and `sadness` (different branches) have high cost

This allows the model to learn that predicting `amusement` when the true label is `joy` is less wrong than predicting `sadness`.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{yu2025emotional,
  title={Emotional Earth Mover’s Distance for Fine-Grained Hierarchical Emotion Analysis},
  author={Yu, Hai-Tao and Li, Dawei and Kang, Xin},
  booktitle={International Conference on Advanced Data Mining and Applications},
  pages={296--310},
  year={2025},
  organization={Springer}
}
```

## References

- [GoEmotions Dataset](https://github.com/google-research/google-research/tree/master/goemotions)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Python Optimal Transport (POT)](https://pythonot.github.io/)

## License

[Apache-2.0](LICENSE). The training code builds on [monologg/GoEmotions-pytorch](https://github.com/monologg/GoEmotions-pytorch) (Apache-2.0); `baselines/bert/` keeps its original license and history.

## Contributing

Issues and pull requests are welcome.
