import torch
import ot
import numpy as np

def hierarchy_path(hierarchy, label):
    path = [label]
    while label in hierarchy:
        label = hierarchy[label]
        path.append(label)
    return path

def hierarchy_distance(hierarchy, label1, label2):
    path1 = hierarchy_path(hierarchy, label1)
    path2 = hierarchy_path(hierarchy, label2)

    common_ancestor_distance = min(path1.index(ancestor) + path2.index(ancestor)
                                   for ancestor in path1 if ancestor in path2)
    return common_ancestor_distance

def compute_cost_matrix(hierarchy, leaf_labels):
    num_labels = len(leaf_labels)
    cost_matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for i in range(num_labels):
        for j in range(i + 1, num_labels):
            distance = hierarchy_distance(hierarchy, leaf_labels[i], leaf_labels[j])
            cost_matrix[i][j] = distance
            cost_matrix[j][i] = distance
    return cost_matrix

hierarchy = {
    "admiration": "joy_lev2",
    "amusement": "joy_lev2",
    "anger_lev3": "anger_lev2",
    "annoyance": "anger_lev2",
    "approval": "joy_lev2",
    "caring": "joy_lev2",
    "confusion": "surprise_lev2",
    "curiosity": "surprise_lev2",
    "desire": "joy_lev2",
    "disappointment": "sadness_lev2",
    "disapproval": "anger_lev2",
    "disgust_lev3": "disgust_lev2",
    "embarrassment": "sadness_lev2",
    "excitement": "joy_lev2",
    "fear_lev3": "fear_lev2",
    "gratitude": "joy_lev2",
    "grief": "sadness_lev2",
    "joy_lev3": "joy_lev2",
    "love": "joy_lev2",
    "nervousness": "fear_lev2",
    "optimism": "joy_lev2",
    "pride": "joy_lev2",
    "realization": "surprise_lev2",
    "relief": "joy_lev2",
    "remorse": "sadness_lev2",
    "sadness_lev3": "sadness_lev2",
    "surprise_lev3": "surprise_lev2",
    "joy_lev2": "positive",
    "anger_lev2": "negative",
    "sadness_lev2": "negative",
    "disgust_lev2": "negative",
    "fear_lev2": "negative",
    "surprise_lev2": "ambiguous",
    "positive": "Root",
    "negative": "Root",
    "ambiguous": "Root",
    "neutral": "Root"
}

leaf_labels = ["admiration", "amusement", "anger_lev3", "surprise_lev3", "annoyance", "approval", "caring", "confusion",
               "curiosity", "desire", "disappointment", "disapproval", "disgust_lev3", "embarrassment", "excitement",
               "fear_lev3", "gratitude", "grief", "joy_lev3", "love", "nervousness", "optimism", "pride", "realization",
               "relief", "remorse", "sadness_lev3", "neutral"]