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
.
├── Fine-grained-emotions-analysis-by-LLM.py  # Main script
├── cost_matrix.py                             # Hierarchical cost matrix computation
├── hierarchy.json                             # Emotion hierarchy structure
├── labels.txt                                 # Emotion label mappings
├── train.tsv                                  # Training dataset (required)
├── dev.tsv                                    # Development dataset (required)
├── test.tsv                                   # Test dataset (required)
└── README.md                                  # This file
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

1. **Clone or download this repository**

2. **Install dependencies** (see Requirements section above)

3. **Configure OpenAI API Key**:
   Edit `Fine-grained-emotions-analysis-by-LLM.py` and replace `'your api key'` on line 33 with your actual OpenAI API key:
   ```python
   client = OpenAI(api_key='your-api-key-here')
   ```
   
   Alternatively, you can set it as an environment variable for better security.

4. **Prepare your datasets**:
   Ensure that `train.tsv`, `dev.tsv`, and `test.tsv` files are present in the project directory.

## Usage

1. **Configure the target dataset**:
   By default, the script processes the development set (`dev_df`). To use the test set, modify line 112:
   ```python
   dataset_to_predict = test_df  # Change from dev_df to test_df
   ```

2. **Adjust batch size** (optional):
   Modify the `batch_size` variable on line 55 to control evaluation batch size (default: 170).

3. **Run the script**:
   ```bash
   python Fine-grained-emotions-analysis-by-LLM.py
   ```

4. **Monitor progress**:
   The script displays a progress bar and prints evaluation metrics after each batch:
   - Accuracy
   - Micro F1 Score
   - Macro F1 Score
   - Weighted F1 Score
   - Hamming Loss
   - EMD (Earth Mover's Distance)

5. **Review outputs**:
   - LLM responses are saved to `gpt_responses.txt`
   - Evaluation metrics are printed to console

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
To use a different OpenAI model (e.g., GPT-4-turbo), modify line 100 in `Fine-grained-emotions-analysis-by-LLM.py`:
```python
response = client.chat.completions.create(model="gpt-4-turbo", ...)
```

### Modifying Emotion Labels
Edit `labels.txt` to add, remove, or modify emotion categories. Ensure that the `hierarchy.json` file is updated accordingly to reflect the hierarchical relationships.

### Adjusting the Prompt
Customize the classification prompt by modifying the `create_prompt()` function (lines 72-95) to change the reasoning approach or examples.

## Notes

- The script processes data sequentially and makes API calls to OpenAI. Processing time depends on dataset size and API rate limits.
- API costs will be incurred based on OpenAI's pricing for the selected model.
- The EMD computation requires the hierarchical structure to compute meaningful distances between emotion categories.

## License

This project is provided as-is for research and educational purposes.

## Contact

For questions or issues, please refer to the project repository.

---
