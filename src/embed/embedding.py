import pandas as pd
import numpy as np
import torch
from embed.Embedding import Embedding
from functions import load_data
from config import DATA_PATH, EMBEDDING_PATH

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Set display width
pd.set_option('display.colheader_justify', 'center')  # Center column headers


# model_name = "meta-llama/Llama-2-7b-chat-hf"
# quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
#                                          bnb_4bit_compute_dtype=torch.bfloat16)



def fuse_embeddings(vector1, vector2, fuse_method):
    tensor1 = torch.tensor(vector1)
    tensor2 = torch.tensor(vector2)

    if method == "concat":
        fused = torch.cat([tensor1, tensor2], dim=1)
    elif method == "sum":
        fused = torch.add(tensor1, tensor2)
    elif method == "mult":
        fused = tensor1 * tensor2
    elif method == "mean":
        fused = torch.mean(torch.stack([tensor1, tensor2], dim=0), dim=0)
    else:
        raise ValueError(f"Unsupported fusion method: {method}")

    return fused

X = load_data(DATA_PATH + '/grouped/X_preprocessed.csv') # synergy_pubchem
print("Loaded the data!")
smiles1 = X['SMILES1'].values #X['smiles_1'].values
smiles2 = X['SMILES2'].values #X['smiles_2'].values
unique_drugs_smiles, indices = np.unique(np.concatenate((smiles1, smiles2)), return_inverse=True)

smiles1 = pd.DataFrame(smiles1, columns=["smiles"])
smiles2 = pd.DataFrame(smiles2, columns=["smiles"])
model_names = [
    # 'chemberta_simcse',
    # 'sbert',
    # 'gpt',
    # 'mol2vec',
    # 'bert',
    'doc2vec',
    # 'angle',
    # 'chemberta_deepchem',
    # 'bert_smiles'
]
fusion_methods = ['sum']
embedding_class = Embedding()
for model_name in model_names:
    print(f"-------- {model_name} --------")
    unique_drugs_embeddings = embedding_class.get_embeddings(model_name, unique_drugs_smiles)
    print("Got the Unique Drug Embeddings!")
    unique_drugs_embeddings['smiles'] = unique_drugs_smiles
    unique_drugs_embeddings.set_index('smiles', inplace=True)
    smiles1_embedding = np.array(smiles1.merge(unique_drugs_embeddings, how='left', left_on='smiles', right_on='smiles').drop(columns=['smiles']))
    smiles2_embedding = np.array(smiles2.merge(unique_drugs_embeddings, how='left', left_on='smiles', right_on='smiles').drop(columns=['smiles']))
    print("Got the SMILES1 and SMILES2 Embeddings!")
    for method in fusion_methods:
        fused_embeddings = fuse_embeddings(smiles1_embedding, smiles2_embedding, fuse_method=method)
        print("Fused the Embeddings!")
        path = EMBEDDING_PATH + f"/drug_pairs-{model_name}_{method}.csv"
        pd.DataFrame(fused_embeddings.detach().numpy()).to_csv(path, sep='\t', index=False)
        print("Saved the Embeddings!")