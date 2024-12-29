from config import CHECKPOINT_PATH, DATA_PATH, LOG_PATH
from embed.Embedding import Embedding
from functions import load_pkl, load_embedding, log
import pandas as pd
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from EarlyStopping import EarlyStopping
from Model import Model
from eval import calculate_metrics
from helpers import set_seed
from helpers import explode_labels, generate_zero_embeddings, construct_hetero_data, link_loader, split_data
from params import settings
from accelerate import Accelerator



set_seed(42)

pdrugs_embeddings = load_embedding('drug_pairs','chemberta_simcse_sum')
pdrugs_embeddings['Drug_ID'] = [i for i in range(0, len(pdrugs_embeddings))]

labels_list = load_pkl(DATA_PATH + 'grouped/labels_list_dict.pkl')

y_ungrouped_df = explode_labels(labels_list)
ungrouped_df = pd.concat([pdrugs_embeddings.reset_index(drop=True), y_ungrouped_df], axis=1).reset_index(drop=True)

unique_ses, mapping_ses = np.unique(np.array(y_ungrouped_df), return_inverse=True)

edges = pd.DataFrame({'src': ungrouped_df['Drug_ID'],'dst': mapping_ses})

labels_embeddings = Embedding().get_embeddings('bert', unique_ses)

pdrugs_zeros = generate_zero_embeddings(pdrugs_embeddings.shape[0], 200)
labels_zeros = generate_zero_embeddings(labels_embeddings.shape[0], 768)

edge_index = torch.tensor(np.array(edges)).t().contiguous()

print("Data Preparation Completed!")
# --------------------------------------------------------


test_aucs = []
test_auprcs = []

accelerator = Accelerator()
device = accelerator.device
print(f"Device: '{device}'")

for iteration in range(settings['model']['gnn']['iter']):

    for model_name in settings['model']['gnn']['model_names']:
        set_seed(42)
        # pdrugs_embeddings = load_embedding('drug_pairs', model_name)
        # pdrugs_embeddings['Drug_ID'] = [i for i in range(0, len(pdrugs_embeddings))]
        # ungrouped_df = pd.concat([pdrugs_embeddings.reset_index(drop=True), y_ungrouped_df], axis=1).reset_index(
        #     drop=True)
        # edges = pd.DataFrame({'src': ungrouped_df['Drug_ID'], 'dst': mapping_ses})
        # pdrugs_zeros = generate_zero_embeddings(pdrugs_embeddings.shape[0], 200)
        # labels_zeros = generate_zero_embeddings(labels_embeddings.shape[0], 768)
        #
        # edge_index = torch.tensor(np.array(edges)).t().contiguous()


        print('\n')
        print(f'======================= {model_name} =======================')
        print('\n')
        if model_name != 'zeros':
            pdrugs_embeddings = load_embedding('drug_pairs', model_name)
            pdrugs = pdrugs_embeddings.iloc[:, :-1]
            labels = labels_embeddings
        else:
            pdrugs = pdrugs_zeros
            labels = labels_zeros


        data = construct_hetero_data(pdrugs, labels, edge_index)

        train_data, val_data, test_data = split_data(data)

        train_loader = link_loader(train_data, batch_size=65536, shuffle=True)
        val_loader = link_loader(val_data, batch_size=2048, shuffle=False)
        test_loader = link_loader(test_data, batch_size=2048, shuffle=False)


        model = Model(data, hidden_channels=settings['model']['gnn']['hidden_channels'], pdrugs_size=pdrugs.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings['model']['gnn']['lr'])

        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        earlystopping = EarlyStopping(patience=2, path=f'{CHECKPOINT_PATH}/checkpoint_{model_name}.pt')
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        for epoch in range(settings['model']['gnn']['epochs']):

            model.train()
            total_loss = total_examples = 0
            train_preds = []
            train_ground_truths = []

            for sampled_data in tqdm.tqdm(train_loader):
                optimizer.zero_grad()

                sampled_data.to(device)
                pred, edge_label = model(sampled_data, is_neg_sampling=True)
                loss = F.binary_cross_entropy_with_logits(pred, edge_label)
                # loss.backward()
                accelerator.backward(loss)
                optimizer.step()

                batch_size = edge_label.size(0)
                total_loss += loss.item() * batch_size
                total_examples += batch_size

                train_preds.append(pred)
                train_ground_truths.append(edge_label)

            (train_auc1, train_auc2), train_auprc, train_ap_at_k = calculate_metrics(train_preds, train_ground_truths)
            print(f"Epoch: {epoch:03d}")
            print(f"Training Loss: {total_loss / total_examples:.4f} | Training AUC: {train_auc1: .4f}, {train_auc2: .4f} | Training AUPRC: {train_auprc: .4f} | Training AP@50: {train_ap_at_k: .4f}")

            model.eval()
            model, val_loader = accelerator.prepare(model, val_loader)
            val_total_loss = 0
            val_total_examples = 0
            val_ground_truths = []
            val_preds = []
            with torch.no_grad():
                for sampled_data in tqdm.tqdm(val_loader):
                    sampled_data.to(device)
                    pred, edge_label = model(sampled_data, is_neg_sampling=True)
                    loss = F.binary_cross_entropy_with_logits(pred, edge_label)

                    batch_size = edge_label.size(0)
                    val_total_loss += loss.item() * batch_size
                    val_total_examples += batch_size

                    val_preds.append(pred)
                    val_ground_truths.append(edge_label)

            (val_auc1, val_auc2), val_auprc, val_ap_at_k = calculate_metrics(val_preds, val_ground_truths)

            print(f"Validation Loss: {val_total_loss / val_total_examples:.4f} | Validation AUC: {val_auc1:.4f}, {val_auc2: .4f} | Validation AUPRC: {val_auprc:.4f} | Loss  Validation AP@50: {val_ap_at_k:.4f}")

            earlystopping(val_auc1, model)
            if earlystopping.early_stop:
                print(f"Early Stopping Triggered at epoch: {epoch}")
                break

        if earlystopping.early_stop:
            model.load_state_dict(torch.load(f'{CHECKPOINT_PATH}/checkpoint_{model_name}.pt'))
            print("Best model weights restored.")

        model.eval()
        model, test_loader = accelerator.prepare(model, test_loader)
        test_ground_truths = []
        test_preds = []

        with torch.no_grad():
            for sampled_data in tqdm.tqdm(test_loader):
                sampled_data.to(device)

                pred, edge_label = model(sampled_data, is_neg_sampling=True)

                test_preds.append(pred)
                test_ground_truths.append(edge_label)


        (test_auc1, test_auc2), test_auprc, test_ap_at_k = calculate_metrics(test_preds, test_ground_truths)

        test_aucs.append(test_auc1)
        test_auprcs.append(test_auprc)

        print(f"Test AUC: {test_auc1:.4f}, {test_auc2: .4f} | Test AUPRC: {test_auprc:.4f} | Test AP@50: {test_ap_at_k:.4f}")

        log(
            ['Model', 'AUC1', 'AUC2', 'AUPRC', 'AP@50'],
            [model_name, float(test_auc1), test_auc2, test_auprc, test_ap_at_k],
            LOG_PATH + f'/gnn_conv_results.csv'
        )

        # test_metrics = [test_auc, test_auprc, test_ap_at_k]
