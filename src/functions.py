from numpy import floating

from config import EMBEDDING_PATH, BASE_PATH, DATA_PATH, PLOT_PATH
import tqdm
import os
import glob
import csv
import numpy as np
import pandas as pd
from typing import List, Any
from sklearn.metrics import average_precision_score
import pickle
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def log(columns: List[Any], values: List[Any], filepath: str) -> None:

    file_exists = os.path.isfile(filepath)

    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(columns)

        writer.writerow(values)
        # for metric_name, metric_value in zip(model.metrics_names, test_metrics):
        #     writer.writerow([model_name, metric_name, metric_value] + values)


def load_embedding(dataset_name: str, model: str) -> pd.DataFrame:

    PATH = None
    for path in glob.iglob(f'{EMBEDDING_PATH}/*'):
        if dataset_name in path and model in path:
            PATH = path
    return pd.read_csv(PATH, sep='\t')

def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    full_path = os.path.join(DATA_PATH, file_path)
    return pd.read_csv(full_path, **kwargs)


def shuffle_df(df: pd.DataFrame, frac=1, random_state=42):
    return df.sample(frac=frac, random_state=random_state).reset_index(drop=True)

def average_precision_at_k_multi_label(y_true: np.ndarray[int], y_pred: np.ndarray[int], k=50) -> floating:
    ap_at_k_list = []

    for i in range(y_true.shape[1]):
        # Sort predictions for the current label by their scores in descending order
        sorted_indices = np.argsort(y_pred[:, i])[::-1][:k]
        sorted_true = y_true.iloc[sorted_indices, i]

        # Calculate average precision at k for the current label
        ap_at_k_label = average_precision_score(sorted_true, y_pred[sorted_indices, i])
        ap_at_k_list.append(ap_at_k_label)

    # Calculate the mean AP@k across all labels
    return np.mean(ap_at_k_list)

def load_pkl(path):
    with open(path, 'rb') as f:
        labels_list = pickle.load(f)

    return labels_list

# def save_auc_across_folds_plot(histories, model_name):
#     train_auc = []
#     val_auc = []
#
#     for i in range(10):
#         train_auc.append(histories[i].history['AUC'][-1])
#         val_auc.append(histories[i].history['val_AUC'][-1])
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_auc, label='Train AUC')
#     plt.plot(val_auc, label='Validation AUC')
#     plt.title('AUC Across Folds')
#     plt.xlabel('Fold')
#     plt.ylabel('AUC')
#     plt.legend()
#
#     plot_filename = os.path.join('model_plots2', f'auc_across_folds_{model_name}.png')
#     plt.savefig(plot_filename)
#     plt.close()  # Close the plot to free up memory
#     print(f"Plot saved as {plot_filename}")


def average_precision_at_k(y_true, y_pred, k=50):
    # Sort predictions in descending order
    sorted_indices = np.argsort(y_pred)[::-1]

    # Get the top-k indices
    top_k_indices = sorted_indices[:k]

    # Get the true labels for the top-k predictions
    top_k_true = y_true[top_k_indices]

    # Compute precision at each rank
    precisions = [
        np.sum(top_k_true[:i + 1]) / (i + 1)  # Precision at rank i
        for i in range(len(top_k_true))
    ]

    # Compute average precision (mean of all precisions at ranks where true label is 1)
    ap_at_k = np.sum(precisions * top_k_true) / np.sum(top_k_true) if np.sum(top_k_true) > 0 else 0

    return ap_at_k

def get_interaction_embeddings(model, test_loader, global_pos_edges, device):
    model.eval()
    plot_ground_truths = []
    plot_preds = []
    plot_probs = []

    with torch.no_grad():
        for sampled_data in tqdm.tqdm(test_loader):
            sampled_data.to(device)

            pred, edge_label, _ = model(sampled_data, global_pos_edges, is_neg_sampling=True, integrate=True)
            prob = torch.sigmoid(pred.sum(-1))

            plot_preds.append(pred)
            plot_probs.append(prob)
            plot_ground_truths.append(edge_label.detach().cpu().numpy())

    em_pred = torch.cat(plot_preds, dim=0).cpu().detach().numpy()
    em_ground_truth = np.concatenate(plot_ground_truths)
    em_prob = torch.cat(plot_probs, dim=0).cpu().detach().numpy()
    return em_pred, em_prob, em_ground_truth

def plot_interaction_embeddings(em_pred, em_prob, em_ground_truth, model_name):
    indices = np.arange(em_pred.shape[0])
    np.random.shuffle(indices)

    em_pred_shuffled = em_pred[indices]
    em_ground_truth_shuffled = em_ground_truth[indices]
    em_prob = em_prob[indices]

    em_pred_subset = em_pred_shuffled[:10000]
    em_ground_truth_subset = em_ground_truth_shuffled[:10000]
    em_prob = em_prob[:10000]

    # Standardize the embeddings
    scaler = StandardScaler()
    em_pred_subset_normalized = scaler.fit_transform(em_pred_subset)

    pca = PCA(
        n_components=32,
        svd_solver='randomized',  # Efficient for large datasets
        random_state=42,  # For reproducibility
        # iterated_power=5,  # Increase for potentially better accuracy, at the cost of speed
        whiten=False  # Set to True if you want to normalize the variance of components
    )

    reduced_data = pca.fit_transform(em_pred_subset_normalized)

    tsne = TSNE(
        n_components=2,
        perplexity=10,
        learning_rate='auto',
        n_iter=300,
        early_exaggeration=20,
        init='pca',
        metric='cosine',  # 'cosine',
        random_state=42,
        n_jobs=-1
    )

    em_pred_2d = tsne.fit_transform(reduced_data)

    plt.figure(figsize=(10, 6))
    # Scatter plot for negative samples (em_ground_truth == 0)
    plt.scatter(em_pred_2d[em_ground_truth_subset == 0, 0], em_pred_2d[em_ground_truth_subset == 0, 1],
                c='red', label='Negative', alpha=0.5, s=10)
    # Scatter plot for positive samples (em_ground_truth_subset == 1)
    plt.scatter(em_pred_2d[em_ground_truth_subset == 1, 0], em_pred_2d[em_ground_truth_subset == 1, 1],
                c='blue', label='Positive', alpha=0.5, s=10)
    plt.title('t-SNE of Interaction Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()

    plt.savefig(PLOT_PATH + f'/IE2_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
