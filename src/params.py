settings = {
    'model':{
        'mlp': {
            'lr': [0.005],
            'n_folds': 10,
            'epochs': 100,
            'k': 50,
            'dropout_rates': [0.2],
            'neurons_per_layer': [[512, 1024, 2048]]
        },
        'gnn': {
            'lr': 0.01,
            'epochs': 30,
            'iter': 3,
            'hidden_channels': 64,
            'model_names': [
                'chembert_simcse_sum',
                # 'deepchem_chemberta10_concat',
                # 'deepchem_chemberta77_concat',
                # 'gpt_sum',
                # 'bert_sum',
                # 'mol2vec_sum',
                # 'doc2vec_sum',
                # 'sbert_sum',
                # 'bert_smiles',
                # 'zeros'
            ]
        }
    }
}