import pandas as pd
from sklearn.preprocessing import LabelEncoder

from functions import log, load_embedding, load_data, shuffle_df, average_precision_at_k_multi_label, load_pkl
from graph.helpers import explode_labels
from mlp.MLPModel import MLPModel
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping
from params import settings

models = [
    # 'chembert_bert_concat',
    # 'chembert_bert_attention1',
    # 'chembert_auto768',
    # 'bert_concat',
    # 'bert_auto1536',
    # 'sbert_concat',
    # 'sbert_auto768',
    # 'mol2vec_concat',
    # 'mol2vec_auto600',
    # 'doc2vec_concat',
    # 'doc2vec_auto100'
    # 'llama_concat',
    # 'llama_auto8192'
    ]

models_agg = [
    # 'chembert_bert_concat',
    'chemberta_simcse_sum',
    # 'chembert_bert_mult',
    # 'chembert_bert_mean',
    # 'bert_concat',
    # 'bert_sum',
    # 'bert_mult',
    # 'bert_mean',
    # 'sbert_concat',
    # 'sbert_sum',
    # 'sbert_mult',
    # 'sbert_mean',
    # 'mol2vec_concat',
    # 'mol2vec_sum',
    # 'mol2vec_mult',
    # 'mol2vec_mean',
    # 'doc2vec_concat',
    # 'doc2vec_sum',
    # 'doc2vec_mult',
    # 'doc2vec_mean',
    # 'gpt_concat',
    # 'gpt_sum',
    # 'gpt_mult',
    # 'gpt_mean',
]


for model_name in models_agg:
    for neurons_per_layer in settings['model']['mlp']['neurons_per_layer']:
        for lr_rate in settings['model']['mlp']['lr']:
            for dr_rate in settings['model']['mlp']['dropout_rates']:
                print("\n")
                print(f"=================       {model_name}      =================")
                print("\n")

                X = shuffle_df(load_embedding('drug_pairs', model_name))
                y = shuffle_df(load_data('labels_hot_preprocessed_pub.csv', sep='\t'))
                X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = 0.1, random_state=1)

                print('\nSizes:')
                print(f'\tOriginal X size: {X.shape}')
                print(f'\ty size: {y.shape}')
                print(f'\tX cross-validation size: {X_train_val.shape}')
                print(f'\tX_test size: {X_test.shape}')


                n_inputs, n_outputs = X_train_val.shape[1], y.shape[1]
                print('\nI/O Model Size:')
                print(f'\t Input Size: {n_inputs}')
                print(f'\t Output Size: {n_outputs}')


                histories = []
                fold_counter = 0
                early_stopping = EarlyStopping(monitor='val_AUC', mode='max', verbose=1)
                kf = KFold(n_splits=settings['model']['mlp']['n_folds'], shuffle=True, random_state=42)

                builder = MLPModel(lr_rate)
                builder.create_architecture(n_inputs, n_outputs, neurons_per_layer, dr_rate)
                model = builder.compile()

                print("\n")
                print("================================================")
                print("                  Training Phase                ")
                print("================================================")
                print("\n")

                for train_index, validation_index in kf.split(X_train_val, y_train_val):

                    print(f'\n\t Fold: {fold_counter + 1} \n')
                    fold_counter = fold_counter + 1

                    X_train, X_validation = X_train_val.iloc[train_index], X_train_val.iloc[validation_index]
                    y_train, y_validation = y_train_val.iloc[train_index], y_train_val.iloc[validation_index]

                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_validation, y_validation),
                        verbose=1,
                        epochs=settings['model']['mlp']['epochs'],
                        callbacks=[early_stopping],
                        workers=4,
                        # batch_size=128
                    )
                    histories.append(history)


                print("\n")
                print("================================================")
                print("                    Test Phase                  ")
                print("================================================")
                print("\n")

                test_metrics = model.evaluate(X_test, y_test, use_multiprocessing=False, verbose=1)
                y_pred = model.predict(X_test, verbose=1)
                k = settings['model']['mlp']['k']
                test_ap_at_k = average_precision_at_k_multi_label(y_test, y_pred, k=k)

                log(
                    columns=['Model','Loss','AUC','AUPRC',f'AP@{k}', 'Comment'],
                    values=[model_name, test_metrics[0], test_metrics[1], test_metrics[2], test_ap_at_k, 'Only Dropout'],
                    filepath='mlp.csv'
                )