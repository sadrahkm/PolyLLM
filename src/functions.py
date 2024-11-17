from numpy import floating

from config import EMBEDDING_PATH, BASE_PATH, DATA_PATH

import os
import glob
import csv
import numpy as np
import pandas as pd
from typing import List, Any
from sklearn.metrics import average_precision_score


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
