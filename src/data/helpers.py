import pandas as pd

def log_unique_ids(df, column1, column2, path, remove_prefix=True):

    unique_drugs = set(pd.concat([df[column1], df[column2]]))
    if remove_prefix:
        unique_drugs = {id.replace('CID', '') for id in unique_drugs}

    with open(path, 'w') as file:
        for i in unique_drugs:
            file.write(str(i) + '\n')

def merge_unique_poly(DDI_graph, cid_smiles_df):
    merged_dataset = pd.merge(DDI_graph, cid_smiles_df, left_on='# STITCH 1', right_on='CID2', how='left')

    # Rename columns for clarity
    merged_dataset.rename(columns={'SMILES': 'SMILES1'}, inplace=True)

    # Merge again for Drug2
    merged_dataset = pd.merge(merged_dataset, cid_smiles_df, left_on='STITCH 2', right_on='CID2', how='left')

    # Rename columns for clarity
    merged_dataset.rename(columns={'SMILES': 'SMILES2'}, inplace=True)

    # Drop the redundant CID columns
    merged_dataset.drop(['CID2_x', 'CID2_y'], axis=1, inplace=True)
    return merged_dataset