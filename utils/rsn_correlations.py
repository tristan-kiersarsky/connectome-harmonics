import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve, mutual_info_score
import scipy


def load_rsns():
    mat = scipy.io.loadmat("data/nhw2022-network-harmonics-data.mat")
    partition, _, labels = mat["yeoLabs"][0][0]
    partition = partition[:, 0]
    labels = labels[0]
    rsns = []
    for rsn_idx in range(len(labels)):
        rsns.append(partition == rsn_idx + 1)

    rsns = np.stack(rsns)
    return rsns, labels


def find_optimal_threshold(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    # Find the F-score for each threshold
    f_scores = (2 * precision * recall) / (precision + recall)
    # Find the index of the threshold that maximizes the F-score
    optimal_idx = np.argmax(f_scores)
    # Find the optimal threshold
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, f_scores.max()


def binary_predictions_from(y_true, y_pred):
    positive_thresh, positive_fscore = find_optimal_threshold(y_true, y_pred)
    negative_thresh, negative_fscore = find_optimal_threshold(y_true, -y_pred)

    if positive_fscore > negative_fscore:
        return (y_pred >= positive_thresh).astype(int)
    else:
        return (y_pred >= negative_thresh).astype(int)


def mutual_information(rsn, harmonic):
    return mutual_info_score(rsn, harmonic > 0)


def f_score(rsn, harmonic):
    binary_prediction = binary_predictions_from(rsn, harmonic)
    return f1_score(rsn, binary_prediction)


def plot_correlations(rsn, harmonics, metric=mutual_information, ax=None, label=None):
    scores = []
    for harmonic in harmonics:
        scores.append(metric(rsn, harmonic))

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.bar(np.arange(len(harmonics)), scores, label=label, alpha=0.5)
    ax.set_ylabel("score")
    ax.set_xlabel("harmonic")
    ax.legend()
