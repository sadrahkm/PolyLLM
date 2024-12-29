import pandas as pd
import numpy as np

from config import DATA_PATH

from data.helpers import log_unique_ids
from data.poly import DDI_graph

df = pd.read_csv(DATA_PATH + '/labeled_triples_m.csv')
#
#
# log_unique_ids(df, 'drug_1', 'drug_2', DATA_PATH + 'unique_ids_synergy.txt')

pubchem_output = pd.read_csv(DATA_PATH + '/pubchem_output_synergy.csv')

cid_smiles = pubchem_output[[' cid', 'isosmiles']]

smiles1 = df.merge(cid_smiles, left_on='drug_1', right_on=' cid', how='left')
smiles2 = df.merge(cid_smiles, left_on='drug_2', right_on=' cid', how='left')

final_df = pd.DataFrame({
    'smiles_1': smiles1['isosmiles'],
    'smiles_2': smiles2['isosmiles'],
    'label': df['label']
})

final_df.to_csv(DATA_PATH + '/synergy_pubchem.csv', index=False, sep='\t')