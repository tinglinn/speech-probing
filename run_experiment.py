import os
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser

import data
import probe
from reporter import Evaluator
import regimen

def run_train_probe(args, probe, dataset, loss, regimen):
    """Trains a structural probe according to args.
    Args:
        args: the global config dictionary built by yaml.
            Describes experiment settings.
        probe: An instance of probe.Probe or subclass.
            Maps hidden states to linguistic quantities.
        dataset: An instance of data.SimpleDataset or subclass.
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

    tags = dataset.tags()
    dev_dataloader = dataset.get_dev_dataloader()

    # both true labels and predictions are per batch
    dev_true_labels = [batch[1] for batch in dev_dataloader]  # batch = (embds, labels)
    # integer labels -> must turn to IOB for evaluation
    dev_predictions = regimen.predict(probe, dev_dataloader)

    # convert predictions of ints to predictions of IOB labels
    codes = ["O", "B-PLACE", "I-PLACE", "B-QUANT", "I-QUANT", "B-WHEN", "I-WHEN",
             "B-ORG", "I-ORG", "B-NORP", "I-NORP", "B-PERSON", "I-PERSON", "B-LAW", "I-LAW"]
    dev_IOB_predictions = []
    dev_IOB_true_labels = []

    for batch_idx, batch_labels in enumerate(dev_true_labels):
        curr_batch_labels = []
        curr_batch_predictions = []
        for idx, int_label in enumerate(batch_labels):
            int_label = int(int_label)
            if int_label != -1:
                curr_batch_labels.append(codes[int_label])
                curr_batch_predictions.append(
                    codes[dev_predictions[batch_idx][idx].argmax()])
        dev_IOB_true_labels.append(curr_batch_labels)
        dev_IOB_predictions.append(curr_batch_predictions)
    print("True labels:", len(dev_IOB_true_labels))
    print("Predictions:", len(dev_IOB_predictions))

    # evaluate
    evaluator = Evaluator(dev_IOB_true_labels, dev_IOB_predictions, tags)
    results, evaluation_agg_entities_type = evaluator.evaluate()

    write_results(args, dev_predictions, dev_true_labels, results, evaluation_agg_entities_type)

def write_results(args, prediction_batchs, true_batches, results, agg_results):  
    """
    Writes results into json files in results directory specified in yaml.
    """
    json.dump([prediction_batch.tolist() for prediction_batch in prediction_batchs], open(
        os.join(args["reporting"]["root"], "dev.predictions"), "w"))
    json.dump([true_batch.tolist() for true_batch in true_batches], open(
        os.join(args["reporting"]["root"], "dev.groundtruth"), "w"))
    json.dump(results, open(os.join(args["reporting"]["root"], "dev.results")), "w")
    json.dump(agg_results, open(
        os.join(args["reporting"]["root"], "dev.results_by_entity")), "w")

def execute_experiment(args, train_probe, report_results):
    """
    Execute an experiment as determined by the configuration
    in args.
    Args:
        train_probe: Boolean whether to train the probe
        report_results: Boolean whether to report results
    """
    class_weights = [3.5 for i in range(15)]  # set class weights -> what works better? need to find out
    class_weights[0] = 1.
    class_weights = torch.tensor(class_weights)
    expt_dataset = data.AudioDataset(args)
    print('Loaded dataset')
    expt_probe = probe.LinearProbe(args)  
    expt_regimen = regimen.ProbeRegimen(args)
    expt_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

    if train_probe:
        print('Training probe...')
        run_train_probe(args, expt_probe, expt_dataset, expt_loss, expt_regimen)
    
    if report_results:
        print('Reporting results of trained probe...')
        run_report_results(args, expt_probe, expt_dataset, expt_regimen)


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

