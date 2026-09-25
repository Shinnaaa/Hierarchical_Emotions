import pandas as pd
from openai import OpenAI
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
from cost_matrix import compute_cost_matrix, hierarchy, leaf_labels
import numpy as np
import ot
from tqdm import tqdm
import json

def normalize_rows(arr):
    row_sums = np.sum(arr, axis=1, keepdims=True)
    non_zero_row_sums = np.where(row_sums == 0, 1, row_sums) 
    uniform_distribution = np.full(arr.shape[1], 1.0 / arr.shape[1])

    normalized_arr = np.where(row_sums == 0, uniform_distribution, arr / non_zero_row_sums)

    return normalized_arr

def emd_compute(labels, preds):
    total_emd = 0
    global M
    labels_normalized = normalize_rows(np.array(labels))
    preds_normalized = normalize_rows(np.array(preds))

    for single_pred, single_label in zip(preds_normalized, labels_normalized):
        emd = ot.emd2(single_pred, single_label, M)
        total_emd += emd

    final_emd = total_emd / len(preds_normalized)
    return final_emd

client = OpenAI(api_key='your api key')
M = compute_cost_matrix(hierarchy, leaf_labels)
M = np.array(M)

def read_labels(file_path):
    with open(file_path, 'r') as file:
        labels = [line.strip() for line in file.readlines()]
    return {label: i for i, label in enumerate(labels)}

labels_mapping = read_labels('labels.txt')
with open("hierarchy.json", "r") as file:
    hierarchical_structure = json.load(file)

def read_dataset(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None)
    df.columns = ['Text', 'Labels', 'Index']  
    return df

train_df = read_dataset('train.tsv')
dev_df = read_dataset('dev.tsv')
test_df = read_dataset('test.tsv')

batch_size = 170
random_seed = 42
shuffled_df = dev_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

def process_multilabels(labels_col):
    def process_label(item):
        if isinstance(item, str):
            return set(map(int, item.split(',')))
        elif isinstance(item, int):
            return {item}
        else:
            raise TypeError(f"Unexpected type for label: {type(item)}")
    return labels_col.apply(process_label)

dev_df['Labels'] = process_multilabels(dev_df['Labels'])
index_to_label = {index: label for label, index in labels_mapping.items()}

def create_prompt(text, labels_detail):
    labels_description = "\n".join([f"{i}: {label}" for i, label in enumerate(labels_detail)])

    prompt = (
        "Analyze the following text using a 'chain of thought' approach for multi-label emotion classification. "
        "Identify key phrases in the text that suggest emotional content. Then, explain how each phrase relates to specific emotions based on the categories provided below. At least return one label, If there is no specific emotion, return the label of neutral"
        "Conclude your analysis with a clear and concise summary of all relevant emotions, formatted as a single line of comma-separated values under the heading 'Final Emotions:'. Ensure that this is the final output with no additional text following it.\n"
        f"Only use the following emotion categories:\n{labels_description}\n\n"
        "Additionally, consider the hierarchical structure provided below to understand the relationships among the emotion categories:\n"
        f"{hierarchical_structure}\n\n"
        "Examples:\n"
        "Text: 'I love you like a love song, baby. And if you know that song, it's now in your head.'\n"
        "Step 1: Identify key phrases - 'love you', 'love song'.\n"
        "Step 2: Relate phrases to emotions - 'love you' and 'love song' suggest strong positive emotions associated with Love.\n"
        "Final Emotions: Love\n\n"
        "Text: 'That was hot!'\n"
        "Step 1: Identify key phrases - 'That was hot!'.\n"
        "Step 2: Relate phrases to emotions - 'That was hot!' suggests Admiration and possibly Neutral.\n"
        "Final Emotions: Admiration, Neutral\n\n"
        f"Text: '{text}'\n"
        "Final Emotions:"
    )

    return prompt

def classify_text(text, labels_detail, output_file):
    try:
        prompt = create_prompt(text, labels_detail)
        response = client.chat.completions.create(model="gpt-3.5-turbo",
                                                  messages=[{"role": "system", "content": prompt}])
        response_text = response.choices[0].message.content

        with open(output_file, 'a', encoding='utf-8') as file:
            file.write(f"Text: {text}\nGPT Response: {response_text}\n\n")

        return response_text
    except Exception as e:
        print(f"Error in classify_text: {e}")
        return "-1"

dataset_to_predict = dev_df  # or test_df

def compute_batch_result(y_np_pred_binary, y_np_true_binary):
    f1_micro = f1_score(y_np_true_binary, y_np_pred_binary, average='micro', zero_division=0)
    f1_macro = f1_score(y_np_true_binary, y_np_pred_binary, average='macro', zero_division=0)
    f1_weighted = f1_score(y_np_true_binary, y_np_pred_binary, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_np_true_binary, y_np_pred_binary)
    hamming_loss_value = hamming_loss(y_np_true_binary, y_np_pred_binary)
    emd = emd_compute(y_np_true_binary, y_np_pred_binary)
    result_dict = {}
    result_dict["f1_micro"] = f1_micro
    result_dict["f1_macro"] = f1_macro
    result_dict["f1_weighted"] = f1_weighted
    result_dict["accuracy"] = accuracy
    result_dict["hamming_loss"] = hamming_loss_value
    result_dict["emd"] = emd

    return result_dict

mlb = MultiLabelBinarizer()
output_file = "gpt_responses.txt"
predicted_results = []
batch_predicted_results = []
batch_true_results = []

def encode_single_labels(predicted_labels, labels_mapping):
    num_samples = 1
    num_labels = len(labels_mapping)

    y_pred_binary = np.zeros((num_samples, num_labels), dtype=int)
    label_set = predicted_labels
    if label_set:  
        for label in label_set.split(','):
            label_name = label.strip().lower() 
            if label_name in labels_mapping:  
                label_index = labels_mapping[label_name]  
                y_pred_binary[0, label_index] = 1 
            else:
                print(f"Label not found in mapping: {label_name}")  

    return y_pred_binary

y_true_binary_all = mlb.fit_transform(dev_df['Labels'])


for index, data in tqdm(dev_df.iterrows()):
    text = data['Text']
    label = data['Labels']
    y_true_binary = y_true_binary_all[index]
    tmp_predicted_result = classify_text(text, labels_mapping, output_file)
    predicted_labels = tmp_predicted_result.split('\n')[-1].replace("Final Emotions: ", "")
    predicted_labels = predicted_labels.lower()
    y_pred_binary = encode_single_labels(predicted_labels, labels_mapping)[0]
    batch_true_results.append(y_true_binary)

    batch_predicted_results.append(y_pred_binary)
    if len(batch_predicted_results) % batch_size == 0 and len(batch_predicted_results) != 0:
        ndarray_predicted_result = np.array(batch_predicted_results)
        ndarray_true_result = np.array(batch_true_results)
        batch_result_dict = compute_batch_result(ndarray_predicted_result, ndarray_true_result)
        predicted_results.append(batch_result_dict)
        batch_predicted_results = []
        batch_true_results = []
        print("accuracy:", batch_result_dict["accuracy"])
        print("Micro F1 Score:", batch_result_dict["f1_micro"])
        print("Macro F1 Score:", batch_result_dict["f1_macro"])
        print("Weighted F1 Score:", batch_result_dict["f1_weighted"])
        print("hamming loss:", batch_result_dict["hamming_loss"])
        print("emd:", batch_result_dict["emd"])
