import pandas as pd
import numpy as np
from config import DATA_PATH
from data.helpers import log_unique_ids

from data.helpers import merge_unique_poly

DDI_graph = pd.read_csv(DATA_PATH + 'dataset/ChChSe-Decagon_polypharmacy.csv')

# Uncomment for the first time
# side_effect_counts = DDI_graph['Side Effect Name'].value_counts()
# log_unique_ids(DDI_graph, '# STITCH 1', 'STITCH 2', DATA_PATH + 'unique_ids_poly.txt', remove_prefix=True)



pubchem_output = pd.read_csv(DATA_PATH + '/pubchem_output_poly.csv')
#
# # Not sure about the following line and have to run them to test
#
cid_smiles_df = pubchem_output[['cid', 'isosmiles']].rename(columns={'cid': 'CID', 'isosmiles': 'SMILES'})


not_available_ids = {
    'CID2': [
        'CID000004856', 'CID000005412', 'CID000003454', 'CID000083786',
        'CID000005212', 'CID005353980', 'CID000060843', 'CID000008953',
        'CID000004645', 'CID000005647', 'CID005282044', 'CID000009052',
        'CID005487301', 'CID000110634', 'CID005381226', 'CID000064147',
        'CID005281007', 'CID005362070', 'CID000002182', 'CID000004200',
        'CID000003043', 'CID000003161', 'CID000003405', 'CID000002818',
        'CID000153941', 'CID005493381', 'CID000151165', 'CID000000143',
        'CID000002022', 'CID000065027', 'CID000004052', 'CID000004585',
        'CID000006691'
    ],
    'SMILES': [
        r'CN1C(=C(NC2=CC=CC=N2)O)C(=O)C3=CC=CC=C3S1(=O)=O',
        r'CC1(C2CC3C(C(=O)C(=C(N)O)C(=O)C3(C(=O)C2=C(C4=C1C=CC=C4O)O)O)N(C)C)O',
        r'C1=NC2=C(N1COC(CO)CO)NC(=NC2=O)N',
        r'C1=NNC2=C1NC=NC2=O',
        r'CCCC1=NN(C2=C1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C',
        r'C1=CC=NC(=C1)NS(=O)(=O)C2=CC=C(C=C2)N/N=C/3\C=CC(=O)C(=C3)C(=O)O',
        r'C1=CC(=CC=C1CCC2=CNC3=C2C(=O)N=C(N3)N)C(=O)N[C@H](CCC(=O)O)C(=O)O',
        r'C(=O)(O)[O-].[Na+]',
        r'CC1(C2C(C3C(C(=O)C(=C(N)O)C(=O)C3(C(=O)C2=C(C4=C1C=CC=C4O)O)O)N(C)C)O)O',
        r'CC(C)C(C(=O)OCCOCN1C=NC2=C1NC(=NC2=O)N)N',
        r'CC(C)(C)NCC(=O)NC1=CC(=C2C[C@H]3C[C@H]4[C@@H](C(=O)/C(=C(\N)/O)/C(=O)[C@]4(C(=O)C3=C(C2=C1O)O)O)N(C)C)N(C)C',
        r'CC(=O)CC(C1=CC=C(C=C1)[N+](=O)[O-])C2=C(OC3=CC=CC=C3C2=O)O',
        r'CCCCCN=C(N)NN/C=C/1\C=NC2=C1C=C(C=C2)CO',
        r'CCCC1=NC(=C2N1NC(=NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)CC)OCC)C',
        r'C[C@H]1/C=C/C=C(\C(=O)NC\2=C(C3=C(C4=C(C(=C3O)C)O[C@@](C4=O)(O/C=C/[C@@H]([C@H]([C@H]([C@@H]([C@@H]([C@@H]([C@H]1O)C)O)C)OC(=O)C)C)OC)C)C(=O)/C2=C/NN5CCN(CC5)C)O)/C',
        r'CC(C)[C@@H](C(=O)OCC(CO)OCN1C=NC2=C1NC(=NC2=O)N)N',
        r'CN(C)NN=C1C(=NC=N1)C(=O)N',
        r'C1=CC(=CC=C1C(=O)NCCC(=O)O)N/N=C/2\C=CC(=O)C(=C2)C(=O)O',
        r'C1C2=C(C=CC(=C2Cl)Cl)NC3=NC(=O)CN31',
        r'CN(C)C1C2CC3CC4=C(C=CC(=C4C(=C3C(=O)C2(C(=O)C(=C(N)O)C1=O)O)O)O)N(C)C',
        r'C1CC(OC1CO)N2C=NC3=C2NC=NC3=O',
        r'CC1C2C(C3C(C(=O)C(=C(N)O)C(=O)C3(C(=O)C2=C(C4=C1C=CC=C4O)O)O)N(C)C)O',
        r'C1=CC(=CC=C1C(=O)NC(CCC(=O)O)C(=O)O)NCC2=CN=C3C(=N2)C(=O)N=C(N3)N',
        r'CN1CCN(CC1)C2=C3C=CC=CC3=NC4=C(N2)C=C(C=C4)Cl',
        r'C=C1[C@H](C[C@@H]([C@H]1CO)O)N2C=NC3=C2NC(=NC3=O)N',
        r'C1=C/C(=C/2\N/C(=C\3/C=CC=CC3=O)/N(N2)C4=CC=C(C=C4)C(=O)O)/C(=O)C=C1',
        r'C[C@H](C1=CC(=CC(=C1)C(F)(F)F)C(F)(F)F)O[C@H]2[C@H](N(CCO2)CC3=NC(=O)NN3)C4=CC=C(C=C4)F',
        r'C1C(N(C2=C(N1)NC(=NC2=O)N)C=O)CNC3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O',
        r'C1=NC2=C(N1COCCO)NC(=NC2=O)N',
        r'CCC[C@]1(CC(=O)C(=C(O1)O)[C@H](CC)C2=CC(=CC=C2)NS(=O)(=O)C3=NC=C(C=C3)C(F)(F)F)CCC4=CC=CC=C4',
        r'CC1=CN=C(S1)NC(=C2C(=O)C3=CC=CC=C3S(=O)(=O)N2C)O',
        r'CC1=CC2=C(NC3=CC=CC=C3N=C2S1)N4CCN(CC4)C',
        r'CC(=O)CC(C1=CC=CC=C1)C2=C(OC3=CC=CC=C3C2=O)O'
    ]
}

not_available_ids_df = pd.DataFrame(not_available_ids)


cid_smiles_df['CID2'] = 'CID' + cid_smiles_df['CID'].astype(str).str.zfill(9)

cid_smiles_df = pd.concat([cid_smiles_df, not_available_ids_df], axis=0)[['CID2', 'SMILES']]

merged_dataset = merge_unique_poly(DDI_graph, cid_smiles_df)

merged_dataset.dropna(inplace=True)
merged_dataset.reset_index(inplace=True, drop=True)

merged_dataset.to_csv(DATA_PATH + 'merged_dataset.csv', index=False)

