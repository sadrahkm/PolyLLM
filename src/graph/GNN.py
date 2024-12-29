from torch_geometric.nn import SAGEConv, to_hetero, GCNConv, GraphConv, GATConv, LayerNorm
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, dropout_rate = 0.3):
        super().__init__()

        self.conv1 = GraphConv(hidden_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.dropout1 = torch.nn.Dropout(0.8)
        self.dropout2 = torch.nn.Dropout(0.8)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.dropout1(x)

        x = self.conv2(x, edge_index)

        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.dropout2(x)

        x = self.conv3(x, edge_index)
        x = self.lin1(x)

        return x