import torch
from torch import Tensor

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_pdrugs: Tensor, x_seffect: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_pdrugs = x_pdrugs[edge_label_index[0]]
        edge_feat_seffect = x_seffect[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_pdrugs * edge_feat_seffect).sum(dim=-1)