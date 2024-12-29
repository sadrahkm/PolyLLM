import torch
from torch import Tensor
from Classifier import Classifier
from GNN import GNN
from helpers import set_seed
from torch_geometric.nn import SAGEConv, to_hetero, GCNConv, GraphConv, GATConv
from torch_geometric.data import HeteroData
from torch_geometric.utils import negative_sampling, batched_negative_sampling, structured_negative_sampling
import numpy as np
import random

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

class Model(torch.nn.Module):
    def __init__(self, data, hidden_channels, pdrugs_size):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for pdrugss and seffects:
        set_seed(42)
        self.seffect_lin = torch.nn.Linear(768, hidden_channels)
        self.pdrugs_lin = torch.nn.Linear(pdrugs_size, hidden_channels)
        self.pdrugs_emb = torch.nn.Embedding(data["pdrugs"].num_nodes, hidden_channels)
        self.seffect_emb = torch.nn.Embedding(data["seffect"].num_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneou s variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.classifier = Classifier()

    def forward(self, data: HeteroData, is_neg_sampling=True, is_training=False) -> Tensor:
        set_seed(42)
        # Best
        x_dict = {
            "pdrugs": self.pdrugs_lin(data["pdrugs"].x.float()) + self.pdrugs_emb(data["pdrugs"].node_id),
            "seffect": self.seffect_lin(data["seffect"].x.float()) + self.seffect_emb(data["seffect"].node_id),
        }

        # x_dict = {
        #   "pdrugs": self.pdrugs_lin(data["pdrugs"].x.float()),
        #   "seffect": self.seffect_lin(data["seffect"].x.float()),
        # }

        # x_dict = {
        #   "pdrugs": data["pdrugs"].x.float(),
        #   "seffect": data["seffect"].x.float(),
        # }

        set_seed(42)
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        if is_neg_sampling:
            torch.manual_seed(42)
            neg_edge_index = negative_sampling(
                edge_index=data["pdrugs", "associated", "seffect"].edge_index,
                num_nodes=(data['pdrugs'].num_nodes, data['seffect'].num_nodes),
                num_neg_samples=data["pdrugs", "associated", "seffect"].edge_label_index.size(1),
                force_undirected=True
            )

            # neg_edge_index = torch.stack((i, k))

            edge_label_index = torch.cat(
                [data["pdrugs", "associated", "seffect"].edge_label_index, neg_edge_index],
                dim=-1,
            )

            edge_label = torch.cat([
                data["pdrugs", "associated", "seffect"].edge_label,
                data["pdrugs", "associated", "seffect"].edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)

            # if is_training:
            # edge_label_index, mask, _ = dropout_node(edge_label_index, p=0.4)
            # edge_label = edge_label[mask]
        else:
            edge_label_index = data["pdrugs", "associated", "seffect"].edge_label_index
            edge_label = data["pdrugs", "associated", "seffect"].edge_label

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types

        pred = self.classifier(
            x_dict["pdrugs"],
            x_dict["seffect"],
            edge_label_index
        )

        return pred, edge_label

    def embed(self, data: HeteroData):
        x_dict = {
            "pdrugs": self.pdrugs_lin(data["pdrugs"].x.float()),
            "seffect": self.seffect_lin(data["seffect"].x.float()),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        edge_label_index = data["pdrugs", "associated", "seffect"].edge_label_index
        edge_feat_pdrugs = x_dict["pdrugs"][edge_label_index[0]]
        edge_feat_seffect = x_dict["seffect"][edge_label_index[1]]
        interaction_embeddings = edge_feat_pdrugs * edge_feat_seffect
        return interaction_embeddings