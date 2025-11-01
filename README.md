# Hierarchical Emotions Classification with BERT and Earth Mover's Distance

A neural approach to fine-grained emotion classification that leverages hierarchical emotion structures through Earth Mover's Distance (EMD) loss. This work improves upon Google's GoEmotions dataset and model by incorporating semantic relationships between emotions through a hierarchical cost matrix.

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
git clone <repository-url>
cd Hierarchical_Emotions-main

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

The `--taxonomy` argument specifies which configuration file to use from the `config/` directory. Available taxonomies include:
- `original`: 28 fine-grained emotions
- `ekman`: Ekman's basic emotions
- `group`: Grouped emotions

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
Hierarchical_Emotions-main/
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

The hierarchical loss function with EMD improves emotion classification by:
1. Better handling of semantically related emotions
2. Reduced penalty for predictions that are hierarchically close to ground truth
3. Improved performance on fine-grained emotion distinctions

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

See `LICENSE` file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
