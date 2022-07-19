"""Classes for training and running inference on probes."""
import os
import sys

from torch import optim
import torch
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class ProbeRegimen:
    def __init__(self, args):
        self.args = args
        self.max_epochs = args['probe_training']['epochs']
        self.params_path = os.path.join(
            args['reporting']['root'], args['probe']['params_path'])

    def set_optimizer(self, probe):
        """Sets the optimizer and scheduler for the training regimen.
    
        Args:
        probe: the probe PyTorch model the optimizer should act on.
        """
        self.optimizer = optim.Adam(probe.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=0)

    def train_until_convergence(self, probe, loss, train_dataset, dev_dataset):
        """
        Training loop for linear probe with given hyperparameters.
        """
        self.set_optimizer(probe)
        min_dev_loss = sys.maxsize
        min_dev_loss_epoch = -1

        for epoch_index in tqdm(range(self.max_epochs), desc='[training]'):
            epoch_train_loss = 0
            epoch_dev_loss = 0
            epoch_train_epoch_count = 0
            epoch_dev_epoch_count = 0
            epoch_train_loss_count = 0
            epoch_dev_loss_count = 0
            
            # train
            for batch in tqdm(train_dataset, desc='[training batch]'):
                probe.train()
                self.optimizer.zero_grad()
                embeddings_batch, label_batch = batch
                count = embeddings_batch.shape[0]  # num words per batch
                predictions = probe(embeddings_batch)
                batch_loss = loss(predictions, label_batch)
                batch_loss.backward()
                epoch_train_loss += batch_loss.detach().cpu().numpy() * count.detach().cpu().numpy()
                epoch_train_epoch_count += 1
                epoch_train_loss_count += count.detach().cpu().numpy()
                self.optimizer.step()
            
            # evaluate on val set
            for batch in tqdm(dev_dataset, desc='[dev batch]'):
                self.optimizer.zero_grad()
                probe.eval()
                embeddings_batch, label_batch = batch
                count = embeddings_batch.shape[0]  # num words per batch
                predictions = probe(embeddings_batch)
                batch_loss = loss(predictions, label_batch)
                epoch_dev_loss += batch_loss.detach().cpu().numpy() * count.detach().cpu().numpy()
                epoch_dev_loss_count += count.detach().cpu().numpy()
                epoch_dev_epoch_count += 1
      
            self.scheduler.step(epoch_dev_loss)
            tqdm.write('[epoch {}] Train loss: {}, Dev loss: {}'.format(epoch_index, epoch_train_loss/epoch_train_loss_count, epoch_dev_loss/epoch_dev_loss_count))
            
            if epoch_dev_loss / epoch_dev_loss_count < min_dev_loss - 0.0001:
                torch.save(probe.state_dict(), self.params_path)
                min_dev_loss = epoch_dev_loss / epoch_dev_loss_count
                min_dev_loss_epoch = epoch_index
                tqdm.write('Saving probe parameters')
            elif min_dev_loss_epoch < epoch_index - 4:
                tqdm.write('Early stopping')
                break 
    
    def predict(self, probe, dataset):
        """ Runs probe to compute predictions on a dataset.
        Args:
        probe: An instance of probe.Probe, transforming model outputs to predictions
        dataset: A pytorch.DataLoader object 
        Returns:
        A list of predictions for each batch in the batches yielded by the dataset
        """
        probe.eval()
        predictions_by_batch = []
        for batch in tqdm(dataset, desc='[predicting]'):
            embeddings, labels = batch
            predictions = probe(embeddings)
            predictions_by_batch.append(predictions.detach().cpu().numpy())
        return predictions_by_batch
