from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from tensorflow.keras import optimizers, losses, metrics
from typing import List

class MLPModel:
    def __init__(self, lr_rate: float):
        self.model = Sequential()
        self.lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )

    def create_architecture(self, n_inputs: int, n_outputs: int, neurons_per_layer: List[int], dropout_rate=0.2):
        self.model.add(Dense(neurons_per_layer[0], input_dim=n_inputs, kernel_initializer='he_normal'))
        self.model.add(LeakyReLU(alpha=0.1))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout_rate))

        for neurons in neurons_per_layer[1:]:
            self.model.add(Dense(neurons, kernel_initializer='he_normal'))
            self.model.add(LeakyReLU(alpha=0.1))
            self.model.add(Dropout(dropout_rate))

        self.model.add(Dense(n_outputs, activation='sigmoid'))

    def compile(self, label_smoothing=0.2):
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.lr_schedule),
            loss=losses.BinaryFocalCrossentropy(label_smoothing=label_smoothing),
            metrics=[
                metrics.AUC(multi_label=True, curve='ROC', name="AUC"),
                metrics.AUC(multi_label=True, curve='PR', name="AUPRC"),
            ],
        )

        return self.model