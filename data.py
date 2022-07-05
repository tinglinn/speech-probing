from collections import defaultdict, Counter, namedtuple
import ast
import os
import csv
import itertools 
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class audioDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.batch_size = args['dataset']['batch_size']
        self.observation_class = self.get_observation_class(self.args['dataset']['observation_fieldnames'])
        self.train_obs, self.dev_obs, self.test_obs = self.read_from_disk()
        self.train_dataset = ObservationIterator(self.train_obs)
        self.dev_dataset = ObservationIterator(self.dev_obs)
        self.test_dataset = ObservationIterator(self.test_obs)
        self.observation_class = self.get_observation_class(self.args['dataset']['observation_fieldnames'])
    
    def get_observation_class(self, fieldnames):
        '''Returns a namedtuple class for a single observation.
        The namedtuple class is constructed to hold all language and annotation
        information for a single sentence or document.
        Args:
        fieldnames: a list of strings corresponding to the information in each observation.
        Returns:
        A namedtuple class; each observation in the dataset will be an instance
        of this class.
        '''
        return namedtuple('Observation', fieldnames)

    def read_from_disk(self):
        '''Reads observations from files as specified by the yaml 
        arguments dictionary and optionally adds pre-constructed embeddings for them.
        Returns:
        A 3-tuple: (train, dev, test) where each element in the
        tuple is a list of Observations for that split of the dataset. 
        '''

        # filepath, timestamp_path, audiopath
        train_file_path = os.path.join(self.args['dataset']['file']['root'],
            self.args['dataset']['file']['train_path'])
        dev_file_path = os.path.join(self.args['dataset']['file']['root'],
            self.args['dataset']['file']['dev_path'])
        test_file_path = os.path.join(self.args['dataset']['file']['root'],
            self.args['dataset']['file']['test_path'])
        
        train_timestamp_path = os.path.join(self.args['dataset']['timestamp']['root'],
            self.args['dataset']['timestamp']['train_path'])
        dev_timestamp_path = os.path.join(self.args['dataset']['timestamp']['root'],
            self.args['dataset']['timestamp']['dev_path'])
        test_timestamp_path = os.path.join(self.args['dataset']['timestamp']['root'],
            self.args['dataset']['timestamp']['test_path'])

        train_audio_path = os.path.join(self.args['dataset']['audio']['root'],
            self.args['dataset']['audio']['train_path'])
        dev_audio_path = os.path.join(self.args['dataset']['audio']['root'],
            self.args['dataset']['audio']['dev_path'])
        test_audio_path = os.path.join(self.args['dataset']['audio']['root'],
            self.args['dataset']['audio']['test_path'])
        
        train_observations = self.generate_observations_from_dataset(train_file_path, train_timestamp_path, train_audio_path)
        dev_observations = self.generate_observations_from_dataset(dev_file_path, dev_timestamp_path, dev_audio_path)
        test_observations = self.generate_observations_from_dataset(test_file_path, test_timestamp_path, test_audio_path)

        train_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
            self.args['dataset']['embeddings']['train_path'])
        dev_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
            self.args['dataset']['embeddings']['dev_path'])
        test_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
            self.args['dataset']['embeddings']['test_path'])
        train_observations = self.optionally_add_embeddings(train_observations, train_embeddings_path)
        dev_observations = self.optionally_add_embeddings(dev_observations, dev_embeddings_path)
        test_observations = self.optionally_add_embeddings(test_observations, test_embeddings_path)
        return train_observations, dev_observations, test_observations

    def generate_observations_from_dataset(self, filepath, timestamp_path, audiopath):
        """
        Generates Observation objects for each audio in a directory.
        Each observation object is a list of (word, audio_embedding, w2v2_embeddings, NER)
        Args: the filesystem path to the conll dataset
        Returns: a list of Observations
        """
        observations = []
        
        file = open(filepath)
        lines = csv.reader(file, delimiter="\t")
            
        for line in lines[1:]: # a single audio
            file_id, transcript, labels = self.get_info_from_SLUE_line(line, "NER")
            audio_file = os.path.join(audiopath, file_id + ".ogg")
            timestamp_file = os.path.join(timestamp_path, file_id + ".csv")
            
            if os.path.isfile(audio_file)==False or os.path.isfile(timestamp_file)==False:
                continue

            word_audio_vectors = self.get_token_audio_vector(audio_file, timestamp_file)
            embeddings = [None for x in range(len(word_audio_vectors))]
            labels = self.convert_raw_label_to_list(transcript, labels)
            IOB_tags = self.IOB_tag_labels(labels)

            observation = self.observation_class(file_id, transcript, word_audio_vectors, labels, IOB_tags, embeddings)
            observations.append(observation)
        return observations
    
    def get_token_audio_vector(self, audio, timestamp):
        """
        returns tuple of (word, token_audio)
        """
        word_audios = []
        waveform, sr = librosa.load(audio, sr=16000)
        timestamps = csv.reader(open(timestamp))
        for line in timestamps:
            type = line[3]
            # word = line[2]
            if type == "words":
                start_frame = int(float(line[0])*16000)
                end_frame = int(float(line[1])*16000)
                word_audio = waveform[start_frame:end_frame]
                word_audios.append(word_audio)
        return word_audios

    def convert_raw_label_to_list(self, transcript, labels):
        """
        returns list of NER labels, one for each word in the transcript
        """
        # NER LABELS are formatted as: i.e.[[LAW, 86, 7], [PLACE, 121, 10]], 
        # with 86 being starting character index, 7 being length of ner phrase
        char_idx = 0
        label_list = []

        for word in transcript:
            if labels != None and len(labels) != 0:
                NER_label = labels[0][0]
                start_idx = int(labels[0][1])
                end_idx = int(labels[0][1]) + int(labels[0][2])
                if char_idx > end_idx:
                    labels.remove(labels[0])
                if char_idx >= start_idx and char_idx <= end_idx:
                    label_list.append(NER_label)
                else:
                    label_list.append("")
            else:
                label_list.append("")
            char_idx += len(word)

        assert len(label_list) == len(transcript)
        return label_list
    
    def IOB_tag_labels(self, labels):
        """
        takes list of per-token NER labels and returns IOB tagging
        """
        IOB = []
        prefix = "I-"
        last_label = ""
        for label in labels:
            if label != last_label and last_label != "":
                prefix = "B-"
    
            if label == "GPE" or label == "LOC":
                IOB.append(prefix + "PLACE")
            if label == "CARDINAL" or label == "MONEY" or label == "ORDINAL" or label == "PERCENT" or label == "QUANTITY":
                IOB.append(prefix + "QUANT")
            if label == "DATE" or label == "TIME":
                IOB.append(prefix + "WHEN")
            if label == "ORG" or label == "NORP" or label == "PERSON" or label == "LAW":
                IOB.append(prefix + label)
            if label == "":
                IOB.append("O")
            
            prefix = "I-"
            last_label = label
        assert len(labels) == len(IOB)
        return IOB
    
    def optionally_add_embeddings(self, observations, pretrained_embeddings_path):
        """Adds pre-computed w2v2 embeddings from disk to Observations."""
        layer_index = self.args['model']['model_layer']
        print('Loading W2V2 Pretrained Embeddings from {}; using layer {}'.format(pretrained_embeddings_path, layer_index))
        embeddings = self.generate_token_embeddings_from_hdf5(observations, pretrained_embeddings_path, layer_index)
        observations = self.add_embeddings_to_observations(observations, embeddings)
        return observations
    
    def add_embeddings_to_observations(self, observations, embeddings):
        '''Adds pre-computed embeddings to Observations.
        Args:
        observations: A list of Observation objects composing a dataset.
        embeddings: A list of pre-computed embeddings in the same order.
        Returns:
        A list of Observations with pre-computed embedding fields.
        '''
        embedded_observations = []
        for observation, embedding in zip(observations, embeddings):
            embedded_observation = self.observation_class(*(observation[:-1]), embedding)
            embedded_observations.append(embedded_observation)
        return embedded_observations
    
    def generate_token_embeddings_from_hdf5(self, observations, filepath, layer_index):
        '''
        Reads pre-computed embeddings from hdf5-formatted file.
        Embeddings should be of the form (word_count, layer_count, feature_count)
        
        Args:
        args: the global yaml-derived experiment config dictionary.
        observations: A list of Observations composing a dataset. 
        filepath: The filepath of a hdf5 file containing embeddings.
        layer_index: The index corresponding to the layer of representation
            to be used. 
        
        Returns:
        A list of numpy matrices; one for each observation.
        Raises:
        AssertionError: word_count of embedding was not the length of the
            corresponding transcript sentence in the dataset.
        '''
        hf = h5py.File(filepath, 'r') 
        single_layer_features_list = []
        for observation in observations:
            file_id = observation.file_id
            feature_stack = hf[file_id]
            single_layer_features = feature_stack[:,layer_index,:]
            assert single_layer_features.shape[0] == len(observation.transcript)
            single_layer_features_list.append(single_layer_features)
        return single_layer_features_list

    def get_info_from_SLUE_line(self, line, task):
        """
        tsv file is formatted as 
        id  raw_text normalized_text speaker_id split raw_ner normalized_ner
        """
        if task == "NER":
            id = line[0]
            transcript = line[2].strip().split()
            labels = ast.literal_eval(line[6])
        return id, transcript, labels
    
    def get_train_dataloader(self, shuffle=True, use_embeddings=True):
        """Returns a PyTorch dataloader over the training dataset.
        Args:
        shuffle: shuffle the order of the dataset.
        use_embeddings: ignored
        Returns:
        torch.DataLoader generating the training dataset (possibly shuffled)
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_wrapper, shuffle=shuffle)

    def get_dev_dataloader(self, use_embeddings=True):
        """Returns a PyTorch dataloader over the development dataset.
        Args:
        use_embeddings: ignored
        Returns:
        torch.DataLoader generating the development dataset
        """
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, collate_fn=self.collate_wrapper, shuffle=False)

    def get_test_dataloader(self, use_embeddings=True):
        """Returns a PyTorch dataloader over the test dataset.
        Args:
        use_embeddings: ignored
        Returns:
        torch.DataLoader generating the test dataset
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_wrapper, shuffle=False)
    
    def collate_wrapper(self, batch):
        return ObservationBatch(batch).embeddings, ObservationBatch(batch).IOB_tags


class ObservationBatch:
    """
    Used for custom follate fn for observation iterator. Pads embeddings and IOB tags.
    """
    def __init__(self, data):
        self.data = [list(obs) for obs in data] 
        file_ids, transcripts, w_a_vs, labels, IOB_tags, embeddings = map(list, zip(*self.data))
        
        # pad embeddings
        embeddings = [torch.tensor(embd) for embd in embeddings]
        self.embeddings = torch.nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        
        # pad labels
        self.labels = list(map(list, zip(*itertools.zip_longest(*labels, fillvalue="-1"))))
        self.IOB_tags = list(map(list, zip(*itertools.zip_longest(*IOB_tags, fillvalue="-1"))))

        assert len(self.IOB_tags[0]) == self.embeddings.shape[1]


class ObservationIterator(Dataset):
    """ List Container for lists of Observations and labels for them.
    Used as the iterator for a PyTorch dataloader.
    """

    def __init__(self, observations):
        self.observations = observations
    
    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        # i should maybe put everything for loading into get item instead
        return self.observations[idx]
  
