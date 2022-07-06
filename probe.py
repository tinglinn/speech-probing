import os
import itertools
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class linearProbe(torch.nn.Module):
    """
    Single layer linear probe.
    """
    def __init__(self, inputSize, outputSize):
        super(linearProbe, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def training_loop(dataloader, model, num_epochs, lr, reg, print_every):
    """
    Training loop for linear probe with given hyperparameters.
    """

    losses = []
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)  # ignore filler value
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)

    total_batches = 0
    total_loss = 0
    total_words = 0

    for i in range(num_epochs):
        print("EPOCH", i)

        for batch in dataloader:
            embeddings, labels = batch

            # run model
            predictions = model(embeddings)

            # compute loss and backpropogate
            loss = loss_fn(predictions, labels)
            loss.backward()

            # update gradients
            optim.step()

            # bookkeeping
            total_batches += 1
            total_loss += loss.data * embeddings.shape[0]
            total_words += embeddings.shape[0]

            if total_batches % print_every == 0:
                print('Loss', (total_loss / total_words).item())
                print('Accuracy', (total_loss / total_words).item())

            losses.append((total_loss / total_words).item())

    return losses