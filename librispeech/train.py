from collections import defaultdict
import numpy as np
import torch

import data as data
import models

from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

import os
import sys
import argparse

from seqeval.metrics import f1_score, classification_report
from seqeval.scheme import IOB2


def eval_tokenwise_correctness(predictions, labels, coords):
    correct = 0
    total = 0
    for p, l in zip(predictions, labels):
        # print(p, l)
        correct += int((p == l).sum())
        total += len(p)
    return correct / total


def eval_ner_f1(predictions, labels, coords):
    def batch_revert(x):
        return [data.code_to_task('ner', y) for y in x]
    predictions = [batch_revert(p) for p in predictions]
    labels = [batch_revert(l) for l in labels]

    new_predictions = defaultdict(lambda: defaultdict(int))
    new_labels = defaultdict(lambda: defaultdict(int))

    for pred, lab, coord in zip(predictions, labels, coords):
        for (p, l, c) in zip(pred, lab, coord):
            sent_idx, word_idx = c
            sent_idx = int(sent_idx)
            word_idx = int(word_idx)
            new_predictions[sent_idx][word_idx] = p
            new_labels[sent_idx][word_idx] = l

    # print("New predictions", new_predictions)
    # print("New labels", new_labels)

    # print("Labels 0", [new_predictions[0][i] for i in range(max(new_predictions[0].keys()))])
    predictions = [[p[i] for i in range(max(p.keys()))]
                   for _, p in sorted(new_predictions.items())]
    labels = [[l[i] for i in range(max(l.keys()))]
              for _, l in sorted(new_labels.items())]
    # print(predictions, labels)

    # print(classification_report(labels, predictions, scheme=IOB2))
    return f1_score(labels, predictions, scheme=IOB2)


# Returns a function which takes two lists of lists of elements, one of predictions and one of labels.
# Returns an evaluation metric.
def get_eval_fn(args):
    task = args.task
    if task == 'upos':
        return eval_tokenwise_correctness
    elif task == 'ner':
        return eval_ner_f1
    raise ValueError(f"Can't find eval function for task {task}.")


def train(args, model, loss_fn, eval_fn, train_dataloader, val_dataloader, layer_label):
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.1, patience=4, min_lr=1e-5)
    max_val_acc = -1.
    for epoch in range(args.num_epochs):
        print("Epoch", epoch)
        acc = []
        optim.zero_grad()
        model.train()

        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        train_preds = []
        train_labels = []
        train_coords = []
        for idx, (embedding_batch, label_batch, sent_idx_batch, word_idx_batch) in pbar:
            optim.zero_grad()
            predictions = model(embedding_batch).squeeze()
            loss = loss_fn(predictions, label_batch)
            loss.backward()
            optim.step()
            train_preds.append(torch.argmax(
                predictions, dim=-1).detach().cpu().numpy())
            train_labels.append(label_batch.cpu().detach().numpy())
            train_coords.append(zip(sent_idx_batch, word_idx_batch))

            pbar.set_description(
                f'lr={next(iter(optim.param_groups))["lr"]:.3e}')

        train_accuracy = eval_fn(train_preds, train_labels, train_coords)
        train_preds.clear()
        train_labels.clear()
        print(f'Train accuracy: {train_accuracy:.3f}')

        model.eval()
        pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        val_preds = []
        val_labels = []
        val_losses = []
        val_coords = []

        with torch.no_grad():
            for idx, (embedding_batch, label_batch, sent_idx_batch, word_idx_batch) in pbar:
                predictions = model(embedding_batch).squeeze()
                loss = loss_fn(predictions, label_batch)
                val_losses.append(loss.detach())
                val_preds.append(torch.argmax(
                    predictions, dim=-1).detach().cpu().numpy())
                val_labels.append(label_batch.cpu().detach().numpy())
                val_coords.append(zip(sent_idx_batch, word_idx_batch))

        mean_val_loss = torch.mean(torch.stack(val_losses)).data.item()
        scheduler.step(mean_val_loss)
        val_losses.clear()

        val_acc = eval_fn(val_preds, val_labels, val_coords)
        print(f'Val accuracy: {val_acc:.3f}')

        if val_acc > max_val_acc:
            print("Saving model...")
            torch.save(model, os.path.join(
                args.savepath, f'model-{layer_label}.pt'))

        max_val_acc = max(max_val_acc, val_acc)
    return max_val_acc


def get_savepath(args):
    savepath = f'probing_results/{args.task}_{args.model}_{args.begin_layer}_{args.end_layer}_{args.max_examples}_{args.num_epochs}'
    os.makedirs(savepath, exist_ok=True)
    return savepath

#def train(args, model, loss_fn, eval_fn, train_dataloader, val_dataloader):


def get_dataset(args, s, data_path):
    if args.contextualization_strategy == 'bert':
        return data.BertEmbeddingDataset(f'{s}.txt', data_path, args.model, task=args.task, end=args.max_examples)
    else:
        return data.Wav2VecEmbeddingDataset(f'{s}.txt', data_path, args.model, task=args.task, end=args.max_examples)


def run_experiment(args):

    savepath = get_savepath(args)
    args.savepath = savepath
    train_dataset = get_dataset(
        args, 'train-clean-100', '/u/nlp/data/LibriSpeech-new-ethanchi/LibriSpeech/train-clean-100/')
    val_dataset = get_dataset(
        args, 'dev-clean', '/u/scr/ethanchi/librispeech/LibriSpeech/dev-clean')

    loss_fn = torch.nn.CrossEntropyLoss()
    eval_fn = get_eval_fn(args)

    for layer in range(args.begin_layer, args.end_layer + 1):
        train_dataset.setlayer(layer)
        val_dataset.setlayer(layer)
        print("Probing layer:", layer)
        train_dataloader = train_dataset.get_dataloader(batch_size=32)
        val_dataloader = val_dataset.get_dataloader(
            batch_size=32, shuffle=False)

        model = models.LinearProbe(
            in_dim=1024, out_dim=len(data.field_to_tags[args.task]))
        max_val_acc = train(args, model, loss_fn, eval_fn,
                            train_dataloader, val_dataloader, layer_label=layer)

        with open(os.path.join(savepath, f'val.acc.layer-{layer}.txt'), 'w') as f:
            f.write(f"{max_val_acc:.4f}\n")


MODELS = ['wav2vec_vox_new', 'wav2vec_vox_10m_new', 'wav2vec_vox_100h_new', 'wav2vec2_vox_960h_new',
          'bert-large-cased', 'bert-large-uncased']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", help="The model whose representations are to be probed")
    parser.add_argument(
        "task", type=str, help="The task to probe for. (xpos, upos, or ner)")
    parser.add_argument("--begin_layer", type=int, default=0,
                        help="The first layer to probe")
    parser.add_argument("--end_layer", type=int, default=24,
                        help="The last layer (inclusive) to probe")
    parser.add_argument("--max_examples", type=int, default=-1,
                        help="The maximum number of utterances to use (default: -1)")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="The number of epochs to train for (default: 20)")
    args = parser.parse_args()

    if args.model not in MODELS:
        print(f"Model {args.model} not found. Choose from: {MODELS}.")
        exit()

    args.contextualization_strategy = 'wav2vec' if 'wav2vec' in args.model else 'bert'

    run_experiment(args)
