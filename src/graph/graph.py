from config import CHECKPOINT_PATH, DATA_PATH, LOG_PATH, PLOT_PATH
from embed.Embedding import Embedding
from functions import load_pkl, load_embedding, log, get_interaction_embeddings, plot_interaction_embeddings
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
from sklearn.metrics import roc_auc_score, average_precision_score

set_seed(12)

pdrugs_embeddings = load_embedding('drug_pairs','chemberta_simcse_sum')
pdrugs_embeddings['Drug_ID'] = [i for i in range(0, len(pdrugs_embeddings))]

labels_list = load_pkl(DATA_PATH + 'grouped/labels_list_dict.pkl')

y_ungrouped_df = explode_labels(labels_list)
ungrouped_df = pd.concat([pdrugs_embeddings, y_ungrouped_df], axis=1).reset_index(drop=True)

unique_ses, mapping_ses = np.unique(np.array(y_ungrouped_df), return_inverse=True)

edges = pd.DataFrame({'src': ungrouped_df['Drug_ID'],'dst': mapping_ses})

labels_embeddings = Embedding().get_embeddings('bert', unique_ses)

pdrugs_zeros = generate_zero_embeddings(pdrugs_embeddings.shape[0], 200)
labels_zeros = generate_zero_embeddings(labels_embeddings.shape[0], 768)

edge_index = torch.tensor(np.array(edges)).t().contiguous()

dangerous_seffects_name = [
    'Mumps',
    'carbuncle',
    'Bleeding',
    'emesis'
]

dangerous_seffects_ids = []

for seffect in dangerous_seffects_name:
    dangerous_seffects_ids.append(np.where(np.array(unique_ses) == seffect)[0][0])

print("Data Preparation Completed!")
# --------------------------------------------------------


test_aucs = []
test_auprcs = []

accelerator = Accelerator()
device = accelerator.device
print(f"Device: '{device}'")

for iteration in range(settings['model']['gnn']['iter']):

    for model_name in settings['model']['gnn']['model_names']:
        set_seed(12)
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

        global_pos_edges = set(map(tuple, data["pdrugs", "associated", "seffect"].edge_index.T.tolist()))

        train_data, val_data, test_data = split_data(data)

        train_loader = link_loader(train_data, batch_size=65536, shuffle=False)
        val_loader = link_loader(val_data, batch_size=2048, shuffle=False)
        test_loader = link_loader(test_data, batch_size=2048, shuffle=False)


        model = Model(data, dangerous_seffects_ids, hidden_channels=settings['model']['gnn']['hidden_channels'], pdrugs_size=pdrugs.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings['model']['gnn']['lr'])

        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        earlystopping = EarlyStopping(patience=2, path=f'{CHECKPOINT_PATH}/checkpoint_{model_name}.pt')
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        for epoch in range(settings['model']['gnn']['epochs']):

            model.train()
            total_loss = total_examples = 0
            train_preds = []
            train_ground_truths = []
# TODO prblem with the mol2vec is that its loss function is very high, while chemberta is not high
            for sampled_data in tqdm.tqdm(train_loader):
                optimizer.zero_grad()

                sampled_data.to(device)
                pred, edge_label, _ = model(sampled_data, global_pos_edges, is_neg_sampling=True, is_training=True)
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
                    pred, edge_label, _ = model(sampled_data, global_pos_edges, is_neg_sampling=True, is_training=True)
                    loss = F.binary_cross_entropy_with_logits(pred, edge_label)

                    batch_size = edge_label.size(0)
                    val_total_loss += loss.item() * batch_size
                    val_total_examples += batch_size

                    val_preds.append(pred)
                    val_ground_truths.append(edge_label)

            (val_auc1, val_auc2), val_auprc, val_ap_at_k = calculate_metrics(val_preds, val_ground_truths)

            print(f"Validation Loss: {val_total_loss / val_total_examples:.4f} | Validation AUC: {val_auc1:.4f}, {val_auc2: .4f} | Validation AUPRC: {val_auprc:.4f} | Loss  Validation AP@50: {val_ap_at_k:.4f}")

            # earlystopping(val_auc1, model)
            earlystopping(val_total_loss / val_total_examples, model)
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
        test_edge_label_index = []
        with torch.no_grad():
            for sampled_data in tqdm.tqdm(test_loader):
                sampled_data.to(device)

                pred, edge_label, edge_label_index = model(sampled_data, global_pos_edges, is_neg_sampling=True, is_training=False)
                pred = torch.sigmoid(pred)
                test_preds.append(pred)
                test_ground_truths.append(edge_label)
                test_edge_label_index.append(edge_label_index)


        (test_auc1, test_auc2), test_auprc, test_ap_at_k = calculate_metrics(test_preds, test_ground_truths)

        test_aucs.append(test_auc1)
        test_auprcs.append(test_auprc)

        print(f"Test AUC: {test_auc1:.4f}, {test_auc2: .4f} | Test AUPRC: {test_auprc:.4f} | Test AP@50: {test_ap_at_k:.4f}")

        log(
            ['Model', 'AUC1', 'AUC2', 'AUPRC', 'AP@50'],
            [model_name, float(test_auc1), test_auc2, test_auprc, test_ap_at_k],
            LOG_PATH + f'/gnn_main_results.csv'
        )

        print("Generating the Plots...")

        # em_pred, em_ground_truth = get_interaction_embeddings(model, test_loader, global_pos_edges, device)

        # plot_interaction_embeddings(em_pred, em_ground_truth, model_name)
        print("Plots Generated!!")



        train_probabilities = []
        train_labels = []
        for index, raw_logits in enumerate(test_preds):
            # Step 1: Get raw predictions (logits) from the model
            probabilities = torch.sigmoid(raw_logits).detach().cpu().numpy()  # Apply sigmoid
            # Step 2: Collect ground-truth labels
            labels = test_ground_truths[index].detach().cpu().numpy()
            # Step 3: Append to the list
            train_probabilities.extend(probabilities)
            train_labels.extend(labels)
        train_probabilities = np.array(train_probabilities)
        train_labels = np.array(train_labels)

        from sklearn.isotonic import IsotonicRegression

        print("Training the Regressor...")

        iso_model = IsotonicRegression(out_of_bounds='clip')
        iso_model.fit(train_probabilities, train_labels)

        print("Trained the Regressor!!")
        def specific_seffect(id):
            result_gt = []
            result_pred = []
            batch_id = 0
            for batch in test_loader:
                mapped_id = np.where(batch['seffect'].node_id.detach().cpu().numpy() == id)[0]
                target_id = np.where(test_edge_label_index[batch_id].detach().cpu().numpy()[1] == mapped_id)
                result_gt.append(test_ground_truths[batch_id][target_id])
                result_pred.append(test_preds[batch_id][target_id])
                batch_id = batch_id + 1
            result_gt = [tensor for tensor in result_gt if tensor.numel() > 0]
            result_pred = [tensor for tensor in result_pred if tensor.numel() > 0]
            result_gt2 = []
            for t in result_gt:
                for item in t:  # Iterate through each element in the tensor
                    result_gt2.append(item.unsqueeze(0))
            result_pred2 = []
            for t in result_pred:
                for item in t:  # Iterate through each element in the tensor
                    result_pred2.append(item.unsqueeze(0))
            result_gt = []
            result_pred = []
            for index, item in enumerate(result_pred2):
                probabilities = torch.sigmoid(torch.tensor(item.detach().cpu().numpy()))
                calibrated_probabilities = iso_model.predict(probabilities)
                # Append results
                result_gt.extend(result_gt2[index])
                result_pred.extend(calibrated_probabilities)

            result_gt = [tensor.item() for tensor in result_gt]
            result_pred = [tensor.item() for tensor in result_pred]
            return result_gt, result_pred


        print("Getting the scores of side effects")
        auc = []
        auprc = []
        for i in range(len(unique_ses)):
            try:
                result_gt, result_pred = specific_seffect(i)
                auc.append(roc_auc_score(result_gt, result_pred))
                auprc.append(average_precision_score(result_gt, result_pred))
            except:
                auc.append(None)
                auprc.append(None)
        print("The scores retrieved! Generating the CSV file...")

        df = pd.DataFrame({
            'Name': unique_ses,
            'AUC Score': auc,
            'AUPRC Score': auprc
        })

        df.to_csv(LOG_PATH + f"/seffect_scores-{model_name}.csv", sep='\t', index=False)
        print(f"CSV file saved for {model_name}")
        print(20 * "_")
