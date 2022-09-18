"""
For running cross-lingual transfer experiments.
"""
import os
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser

import data_cl as data
import probe
import regimen as regimen
from seqeval.metrics import f1_score, classification_report
from seqeval.scheme import IOB2


def get_loss_fn(args):
    task = args["task"]
    if task == "ner":
        class_weight = torch.tensor([5. for i in range(9)])
        return nn.CrossEntropyLoss(weight=class_weight)
    elif task == "upos":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Can't find loss function for task {task}.")


def run_train_probe(args, probe, dataset, loss, regimen):
    """Trains a structural probe according to args.
    Args:
        args: the global config dictionary built by yaml.
            Describes experiment settings.
        probe: An instance of probe.LinearProbe.
            Maps hidden states to linguistic quantities.
        dataset: An instance of data.AudioDataset.
            Provides access to DataLoaders of corpora. 
        reporter: An instance of reporter.Evaluator
            Implements evaluation scripts.
    Returns:
        None; causes probe parameters to be written to disk.
    """
    regimen.train_until_convergence(probe, loss,
                                    dataset.get_train_dataloader(), dataset.get_dev_dataloader())


def run_report_results(args, probe, dataset, regimen):
    """
    Reports results from a structural probe according to args.
    By default, does so only for dev set.
    """
    probe_params_path = os.path.join(
        args['reporting']['root'], args['probe']['params_path'])
    probe.load_state_dict(torch.load(probe_params_path))
    probe.eval()

    dev_dataloader = dataset.get_dev_dataloader()

    # both true labels and predictions are per batch
    true_labels = [batch[1]
                   for batch in dev_dataloader]  # batch = (embds, labels)
    predictions = regimen.predict(probe, dev_dataloader)
    predictions = [np.argmax(prediction, axis=1) for prediction in predictions]

    task = args["task"]
    result = eval_by_task(task, true_labels, predictions)
    write_result(args, result)


def eval_by_task(task, true_labels, predictions):
    if task == "ner":
        codes = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG",
                 "B-PER", "I-PER", "B-MISC", "I-MISC"]
        IOB_true_labels, IOB_predictions = [], []

        for batch_labels, batch_predictions in zip(true_labels, predictions):
            curr_batch_labels = []
            curr_batch_predictions = []
            for label, prediction in zip(batch_labels, batch_predictions):
                curr_batch_labels.append(codes[int(label)])
                curr_batch_predictions.append(codes[prediction])
            assert len(curr_batch_labels) == len(curr_batch_predictions)
            IOB_true_labels.append(curr_batch_labels)
            IOB_predictions.append(curr_batch_predictions)

        return f1_score(IOB_true_labels, IOB_predictions, scheme=IOB2)

    elif task == "upos":
        num_total, num_correct = 0, 0
        for batch_labels, batch_predictions in zip(true_labels, predictions):
            for label, prediction in zip(batch_labels, batch_predictions):
                num_total += 1
                if label == predictions:
                    num_correct += 1
        return num_correct / num_total

    raise ValueError(f"Can't find eval function for task {task}.")


def write_result(args, result):
    layer = args["model"]["model_layer"]
    with open(os.join(args["reporting"]["root"], f"val.result.layer-{layer}.txt"), "w") as f:
        f.write(f"{result:.4f}\n")


def execute_experiment(args, train_probe, report_results):
    """
    Execute an experiment as determined by the configuration
    in args.
    Args:
        train_probe: Boolean whether to train the probe
        report_results: Boolean whether to report results
    """
    expt_dataset = data.AudioDataset(args, "train")
    eval_dataset = data.AudioDataset(args, "eval")
    print('Loaded dataset')
    expt_probe = probe.LinearProbe(args)
    expt_regimen = regimen.ProbeRegimen(args)
    expt_loss = get_loss_fn(args)

    if train_probe:
        print('Training probe...')
        run_train_probe(args, expt_probe, expt_dataset,
                        expt_loss, expt_regimen)

    if report_results:
        print('Reporting results of trained probe...')
        run_report_results(args, expt_probe, eval_dataset, expt_regimen)


if __name__ == '__main__':
    argp = ArgumentParser()
    argp.add_argument('experiment_config')
    argp.add_argument('--train-probe', default=1, type=int,
                      help='Set to train a new probe.; ')
    argp.add_argument('--report-results', default=1, type=int,
                      help='Set to report results; '
                      '(optionally after training a new probe)')

    cli_args = argp.parse_args()

    yaml_args = yaml.safe_load(open(cli_args.experiment_config))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yaml_args['device'] = device

    execute_experiment(yaml_args, train_probe=cli_args.train_probe,
                       report_results=cli_args.report_results)
