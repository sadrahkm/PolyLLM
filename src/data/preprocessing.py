import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import MultiLabelBinarizer

from config import DATA_PATH

merged_dataset = pd.read_csv(DATA_PATH + 'merged_dataset.csv')
merged_dataset = merged_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

side_effect_counts = merged_dataset['Side Effect Name'].value_counts()
print("\n ** Before Filtering the Side Effects: ** \n")
print(side_effect_counts)

merged_dataset = merged_dataset[merged_dataset['Side Effect Name'].isin(side_effect_counts[side_effect_counts > 495].index)]
print("\n -------------------------------- \n")
print("\n ** After Filtering the Side Effects, the ones that have occured more than 500 times ** \n")
print(merged_dataset['Side Effect Name'].value_counts())

grouped_df = merged_dataset.groupby(['# STITCH 1', 'STITCH 2', 'SMILES1', 'SMILES2'])['Side Effect Name'].agg(list).reset_index()
grouped_df.to_csv(DATA_PATH + 'X_preprocessed.csv', index=False)

unique_drugs_smiles = pd.concat([grouped_df['SMILES1'], grouped_df['SMILES2']]).unique()






# # STATS
# print("--------------   STATS   -------------------")
#
# side_effect_counts = merged_dataset['Side Effect Name'].value_counts()
#
# # Sort side effects by frequency (already sorted by default with value_counts)
# sorted_side_effects = side_effect_counts.sort_values(ascending=False)
#
# # Optionally, take the top 20 or 50 side effects to make the plot clearer
# top_side_effects = sorted_side_effects.head(50)
#
# # Plotting the frequencies
# plt.figure(figsize=(12, 8))
# plt.barh(top_side_effects.index, top_side_effects.values, color='skyblue')
# plt.xlabel('Frequency')
# plt.ylabel('Side Effect')
# plt.title('Top 50 Side Effects by Frequency')
# plt.gca().invert_yaxis()  # To have the most common side effect at the top
# plt.show()
#
# unique_drugs = pd.concat([grouped_df['# STITCH 1'], grouped_df['STITCH 2']]).unique()
# print(f"\n ** There are {unique_drugs.shape[0]} unique drugs in the dataset ** \n")
#
#
# mean_length = pd.Series(unique_drugs_smiles).apply(len).mean()
#
# # Plotting the distribution
# plt.figure(figsize=(8, 6))
# sns.histplot(pd.Series(unique_drugs_smiles).apply(len), kde=True, color='skyblue', bins=30)
# plt.axvline(x=mean_length, color='red', linestyle='--', label=f'Mean Length: {mean_length:.2f}')
# plt.title('Distribution of Mean Length of Strings')
# plt.xlabel('Length of Strings')
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()
#
# grouped_df['Side Effect Length'] = grouped_df['Side Effect Name'].apply(len)
# plt.figure(figsize=(10, 6))
# sns.histplot(grouped_df['Side Effect Length'], bins=30, kde=True)
# plt.xlabel('Number of Side Effects')
# plt.ylabel('Number of Drug Pairs')
# plt.show()
#
# print("--------------  END OF STATS   -------------------")







# Labels
grouped_df['Aggregated_Labels'] = grouped_df['Side Effect Name'].apply(lambda x: ', '.join(x))

labels_dict = {tuple(row[['# STITCH 1', 'STITCH 2']]): set(row['Side Effect Name']) for _, row in grouped_df.iterrows()}
labels_list = list(labels_dict.values())
with open(DATA_PATH + 'grouped/labels_list_dict.pkl', 'wb') as f:
    pickle.dump(labels_list, f)


# def load_labels(path):
#     with open(path, 'rb') as f:
#         labels_list = pickle.load(f)
#
#     return labels_list
# labels_list = load_labels('labels_list_dict.pkl')




# MultiLabel
mlb = MultiLabelBinarizer()
mlb_result = mlb.fit_transform(labels_list)
class_labels = mlb.classes_
class_indices = np.arange(len(class_labels))

# Create a mapping from class labels to integer indices
label_to_index = {label: index for label, index in zip(class_labels, class_indices)}
multihot_encoded_labels = pd.DataFrame(mlb_result)
multihot_encoded_labels.to_csv(DATA_PATH + 'grouped/labels_hot_preprocessed.csv', index=False, sep='\t')