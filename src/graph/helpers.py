import pandas as pd
import numpy as np
import torch
import random
from torch_geometric import seed_everything
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# Usage example:
set_seed(12)

def explode_labels(labels_list):
    labels_list = [sorted(lst) for lst in labels_list]
    exploded_labels = pd.Series(labels_list).explode()
    exploded_labels_df = exploded_labels.to_frame()
    exploded_labels_df.columns = ['label']

    return exploded_labels_df


def get_unique_labels(y):
    unique_side_effects = y.drop_duplicates()
    print("Number of Unique Side Effects: ", unique_side_effects.shape)

    side_effect_mapping = {}
    for idx, row in enumerate(unique_side_effects):
        side_effect_mapping[row] = idx

    side_effect_mapping

    side_effects = pd.DataFrame(side_effect_mapping.keys(), columns=['side_effect_name'])
    side_effects.index = side_effect_mapping.values()

    return side_effects, side_effect_mapping


def generate_zero_embeddings(x, y):
    return pd.DataFrame(np.zeros((x, y)))

def construct_hetero_data(items, labels, edge_index, undirected=True):
    data = HeteroData()

    # Save node indices:
    data["seffect"].node_id = np.array(labels.index)
    data["pdrugs"].node_id = np.array(items.index)

    # Add the node features and edge indices:
    data["pdrugs"].x = torch.tensor(np.array(items))
    data['seffect'].x = torch.tensor(np.array(labels))

    data["pdrugs", "associated", "seffect"].edge_index = edge_index

    if undirected:
        data = T.ToUndirected()(data)

    return data

def split_data(data):
    return T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=0.0,
        add_negative_train_samples=False,
        edge_types=("pdrugs", "associated", "seffect"),
        rev_edge_types=("seffect", "rev_associated", "pdrugs"),
    )(data)


def link_loader(train_data, batch_size=65536, shuffle=True, seed=42):
    edge_label_index = train_data["pdrugs", "associated", "seffect"].edge_label_index
    edge_label = train_data["pdrugs", "associated", "seffect"].edge_label

    def worker_init_fn(worker_id):
        """Ensure workers have different seeds."""
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    # Set the global seed for reproducibility
    set_seed(seed)

    return LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        edge_label_index=(("pdrugs", "associated", "seffect"), edge_label_index),
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=worker_init_fn
    )