import torch
from Classifier import Classifier
from GNN import GNN
from helpers import set_seed
from torch_geometric.nn import to_hetero
from torch_geometric.data import HeteroData
from torch_geometric.utils import negative_sampling
import numpy as np


def manual_negative_sampling(data, node_type, id, num_neg_samples=10):
    # Retrieve mapped ID in the subgraph
    mapped_id = np.where(data[node_type].node_id.detach().cpu().numpy() == id)[0][0]

    # Get the edge_label_index for the specific relationship
    edge_label_index = data["pdrugs", "associated", "seffect"].edge_label_index
    drug_nodes = edge_label_index[0].detach().cpu().numpy()  # Drug node IDs
    target_nodes = edge_label_index[1].detach().cpu().numpy()  # Target node IDs

    # Collect all connected drug nodes for the specific mapped ID
    connected_drugs = drug_nodes[target_nodes == mapped_id]

    # Generate random drug node IDs
    all_drugs = np.arange(data["pdrugs"].num_nodes)
    non_connected_drugs = np.setdiff1d(all_drugs, connected_drugs)

    if len(non_connected_drugs) < num_neg_samples:
        raise ValueError("Not enough non-connected drug nodes to sample from.")

    neg_drugs = np.random.choice(non_connected_drugs, size=num_neg_samples, replace=False)

    # Create negative edges (drug_id, target_node)
    neg_edge_index = torch.tensor(
        [neg_drugs, np.full(num_neg_samples, mapped_id)],
        dtype=torch.long,
        device=edge_label_index.device,
    )

    return neg_edge_index

def valid_negative_sampling(data, global_pos_edges, ids, is_training):

    neg_edge_index = negative_sampling(
        edge_index=data["pdrugs", "associated", "seffect"].edge_index,
        num_nodes=(data['pdrugs'].num_nodes, data['seffect'].num_nodes),
        num_neg_samples=data["pdrugs", "associated", "seffect"].edge_label_index.size(1),
        force_undirected=True
    )

    # if not is_training:
    #     for id in ids:
    #         man_neg_edge_index = manual_negative_sampling(data, 'seffect', id, 1)
    #
    #         neg_edge_index = torch.cat(
    #             [neg_edge_index, man_neg_edge_index],
    #             dim=-1,
    #         )
    #         print(man_neg_edge_index)

    neg_edges = set(map(tuple, neg_edge_index.T.tolist()))

    valid_neg_edges = neg_edges - global_pos_edges


    return torch.tensor(list(valid_neg_edges)).T

    # if len(valid_neg_edges) >= num_neg_samples:
    #     # Return the required number of negatives
    #     valid_neg_edges = list(valid_neg_edges)[:num_neg_samples]

class Model(torch.nn.Module):
    def __init__(self, data, dangerous_seffects_ids, hidden_channels, pdrugs_size):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for pdrugss and seffects:
        set_seed(12)
        self.seffect_lin = torch.nn.Linear(768, hidden_channels)
        self.pdrugs_lin = torch.nn.Linear(pdrugs_size, hidden_channels)
        self.pdrugs_emb = torch.nn.Embedding(data["pdrugs"].num_nodes, hidden_channels)
        self.seffect_emb = torch.nn.Embedding(data["seffect"].num_nodes, hidden_channels)
        self.dangerous_seffects_ids = dangerous_seffects_ids

        # instantiate homogeneous GNN
        self.gnn = GNN(hidden_channels)

        # convert GNN model into a heterogeneou s variant
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.classifier = Classifier()

    def forward(self, data: HeteroData, global_pos_edges, is_neg_sampling=True, integrate=False, is_training=True) -> tuple:
        set_seed(12)
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

        set_seed(12)
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        if is_neg_sampling:
            torch.manual_seed(42)

            # neg_edge_index = negative_sampling(
            #     edge_index=data["pdrugs", "associated", "seffect"].edge_index,
            #     num_nodes=(data['pdrugs'].num_nodes, data['seffect'].num_nodes),
            #     num_neg_samples=data["pdrugs", "associated", "seffect"].edge_label_index.size(1),
            #     force_undirected=True
            # )

            neg_edge_index = valid_negative_sampling(
                data=data,
                global_pos_edges=global_pos_edges,
                ids=self.dangerous_seffects_ids,
                is_training=is_training
            )


            neg_edge_index = neg_edge_index.to('cuda')

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
            edge_label_index,
            integrate
        )

        return pred, edge_label, edge_label_index