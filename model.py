import numpy as np
from passage.utils import save, load
from passage.layers import Embedding, GatedRecurrent, Dense, OneHot, LstmRecurrent, Generic
from passage.models import RNN
import random
import sys

def build_model(weights=None, embedding_size=256, recurrent_gate_size=512, n_features=5, dropout=0.4):
    """
    build_model

    Inputs:
        weights - Path to a weights file to load, or None if the model should be built from scratch
        embedding_size - Size of the embedding layer
        recurrent_gate_size - Size of the gated recurrent layer
        n_features - Number of features for the embedding layer
        dropout - Dropout value

    Returns:
        A model object ready for training (or evaluation if a previous model was loaded via `weights`)
    """
    # vvvvv
    #Modify this if you want to change the structure of the network!
    # ^^^^^
    model_layers = [
        Embedding(size=embedding_size,n_features=n_features),
        GatedRecurrent(size=recurrent_gate_size, p_drop=dropout),
        Dense(size=1, activation='sigmoid', p_drop=dropout)
    ]
    model = RNN(layers=model_layers, cost='BinaryCrossEntropy', verbose=2, updater='Adam')
    if weights: #Just load the provided model instead, I guess?
        model = load(weights)
    return model


def train_model(model, train_data, epochs, save_name):
    """
    train_model

    Inputs:
        model - Model object to train
        train_data - Dataset to use during training
        epochs - Number of epochs to train for
        save_name - Prefix for output checkpoint models
    """
    #TODO make sure we are still keeping track of transcript names

    positive, negative = train_data
    #Add explicit labels to positive/negative datasets so we
    #can concat them together without losing info
    positive = label_data(positive, 1)
    negative = label_data(negative, 0)
    all_data = positive+negative

    tokens, labels = zip(*all_data)
    model.fit(tokens, labels, n_epochs=epochs, path=save_name, snapshot_freq=3)

def test():
    model_name = "resources/DeepLincM5.pkl.90"
    from tests import load_test_seqs
    from passage.utils import load
    from evaluate import batch_predict

    seqs = load_test_seqs()
    mrnn = load(model_name)
    for p in batch_predict([mrnn], seqs):
        print p
    

if __name__ == "__main__":
    test()
