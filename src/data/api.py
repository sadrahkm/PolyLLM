import pandas as pd
import numpy as np

from configs import DATA_PATH

df = pd.read_csv(DATA_PATH + '/labeled_triples_m.csv')

unique_drugs = set(pd.concat([df['drug_1'], df['drug_2']]))

with open('unique_ids.txt', 'w') as file:
    for i in unique_drugs:
        file.write(str(i) + '\n')

pubchem_output = pd.read_csv('pubchem_output.csv')

cid_smiles = pubchem_output[[' cid', 'isosmiles']]

smiles1 = df.merge(cid_smiles, left_on='drug_1', right_on=' cid', how='left')
smiles2 = df.merge(cid_smiles, left_on='drug_2', right_on=' cid', how='left')

final_df = pd.DataFrame({
    'smiles_1': smiles1['isosmiles'],
    'smiles_2': smiles2['isosmiles'],
    'label': df['label']
})

final_df.to_csv(DATA_PATH + '/synergy.csv', index=False, sep='\t')
