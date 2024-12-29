import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation AUC doesn't improve after a given patience, and optionally restores the best weights."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print,
                 restore_best_weights=True):
        """
        Args:
            patience (int): How long to wait after last time validation AUC improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation AUC improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
            restore_best_weights (bool): If True, will restore model to the state with the best validation score at the end of training.
                                         Default: True
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = -np.Inf  # Initialize with the worst possible AUC
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.restore_best_weights = restore_best_weights
        self.best_model_state = None

    def __call__(self, val_auc, model):
        score = val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model)
            self.counter = 0

    def save_checkpoint(self, val_auc, model):
        '''Saves model when validation AUC increases.'''
        if self.verbose:
            self.trace_func(f'Validation AUC increased ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
        # Save model state
        # self.best_model_state = model.state_dict().copy()
        # Optionally, save to disk
        torch.save(model.state_dict(), self.path)
        self.val_auc_max = val_auc

    # def get_best_model_state(self):
    #     '''Returns the best model state dict.'''
    #     return self.best_model_state