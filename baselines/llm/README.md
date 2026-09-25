# Fine-grained Emotions Analysis by LLM

A comprehensive multi-label emotion classification system that leverages Large Language Models (LLMs) for fine-grained emotion detection using hierarchical emotion structures. This project implements a chain-of-thought approach to analyze emotional content in text and provides robust evaluation metrics including Earth Mover's Distance (EMD) for hierarchical emotion assessment.

## Features

- **Multi-label Classification**: Supports simultaneous detection of multiple emotions in a single text
- **Hierarchical Emotion Structure**: Utilizes a hierarchical taxonomy to capture relationships between emotion categories
- **Chain-of-Thought Analysis**: Employs a reasoning-based approach for emotion identification
- **Comprehensive Evaluation**: Implements multiple metrics including:
  - Micro, Macro, and Weighted F1 Scores
  - Accuracy
  - Hamming Loss
  - Earth Mover's Distance (EMD) with hierarchical cost matrix
- **Batch Processing**: Efficient batch-wise evaluation with configurable batch sizes
- **LLM Integration**: Supports OpenAI GPT models (currently configured for GPT-3.5-turbo)

## Requirements

### Python Version
Python 3.7 or higher

### Dependencies
Install the required packages using pip:

```bash
pip install pandas openai scikit-learn numpy pot tqdm
```

Required packages:
- `pandas`: Data manipulation and CSV/TSV file handling
- `openai`: OpenAI API client for LLM integration
- `scikit-learn`: Machine learning metrics and preprocessing
- `numpy`: Numerical computations
- `pot`: Python Optimal Transport library for EMD calculations
- `tqdm`: Progress bar visualization

## Project Structure

```
baselines/llm/
├── Fine-grained-emotions-analysis-by-LLM.py  # Main script
├── cost_matrix.py                             # Hierarchical cost matrix computation
├── hierarchy.json                             # Emotion hierarchy given to the model in the prompt
└── README.md                                  # This file

../../data/original/                           # Shared with the rest of the repository
├── labels.txt, train.tsv, dev.tsv, test.tsv
```

## Dataset Format

The dataset files (`train.tsv`, `dev.tsv`, `test.tsv`) should be tab-separated files with the following structure:
- Column 1: Text (the input text to analyze)
- Column 2: Labels (comma-separated label indices, e.g., "0,5,12")
- Column 3: Index (sample index)

Example:
```
I feel great today!	0,18	1
This is disappointing	10	2
```

## Setup

This folder was merged from the former `Shinnaaa/Fine-grained-emotions-analysis-by-LLM` repository with its history. It reads the shared GoEmotions split in `../../data/original/`, so no data needs to be copied.

1. Install the dependencies (see Requirements above).
2. Set your API key: `export OPENAI_API_KEY=sk-...`
   Optional: `export OPENAI_MODEL=...` (default `gpt-3.5-turbo`, as used in the paper) and `OPENAI_BASE_URL` for another OpenAI-compatible endpoint.

## Usage

```bash
python baselines/llm/Fine-grained-emotions-analysis-by-LLM.py   # from any directory
```

- The script classifies every example of the **dev** split (5,426 texts, one API call each) and prints the metrics after every `batch_size` (170) examples: accuracy, micro/macro/weighted F1, Hamming loss and EMD.
- Raw model answers are appended to `gpt_responses.txt` next to the script.
- To evaluate the test split instead, replace `dev_df` with `test_df` in the evaluation loop and in the `mlb.fit_transform(...)` line near the end (the `dataset_to_predict` variable is not used by the loop).

## Evaluation Metrics

- **F1 Scores**: Measures precision and recall balance across labels
  - Micro: Calculates metrics globally by counting total true positives, false negatives, and false positives
  - Macro: Calculates metrics for each label and finds their unweighted mean
  - Weighted: Calculates metrics for each label and finds their average weighted by support

- **Accuracy**: Percentage of correctly classified instances

- **Hamming Loss**: Fraction of labels that are incorrectly predicted

- **Earth Mover's Distance (EMD)**: Computes the minimum cost to transform one probability distribution into another, using a hierarchical cost matrix that accounts for semantic relationships between emotions in the taxonomy

## Customization

### Changing the LLM Model
Set the `OPENAI_MODEL` environment variable, e.g. `export OPENAI_MODEL=gpt-4o`. With `OPENAI_BASE_URL` you can also point the script at any OpenAI-compatible provider.

### Modifying Emotion Labels
Edit `../../data/original/labels.txt` to add, remove, or modify emotion categories (this changes them for the whole repository). Ensure that the `hierarchy.json` file is updated accordingly to reflect the hierarchical relationships.

### Adjusting the Prompt
Customize the classification prompt by modifying the `create_prompt()` function to change the reasoning approach or examples.

## Notes

- The script processes data sequentially and makes API calls to OpenAI. Processing time depends on dataset size and API rate limits.
- API costs will be incurred based on OpenAI's pricing for the selected model.
- The EMD computation requires the hierarchical structure to compute meaningful distances between emotion categories.

## License

This project is provided as-is for research and educational purposes.

## Contact

For questions or issues, please refer to the project repository.

---
