from collections import namedtuple
import ast
import os
import csv
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers.utils.dummy_pt_objects import Wav2Vec2PreTrainedModel
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Tokenizer

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        config = Wav2Vec2Config.from_pretrained(
            "facebook/wav2vec2-large-lv60", output_hidden_states=True)
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-lv60", config=config)
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(
            "facebook/wav2vec2-large-lv60")

        self.args = args
        self.batch_size = args['dataset']['batch_size']
        self.observation_class = self.get_observation_class(
            self.args['dataset']['observation_fieldnames'])
        self.train_obs, self.dev_obs = self.read_from_disk()  # commented out self.test_obs
        self.train_dataset = ObservationIterator(self.train_obs)
        self.dev_dataset = ObservationIterator(self.dev_obs)
        print("Length of dev dataset", len(self.dev_dataset))
        #self.test_dataset = ObservationIterator(self.test_obs)
        self.observation_class = self.get_observation_class(
            self.args['dataset']['observation_fieldnames'])

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

        train_file_path = os.path.join(self.args['dataset']['file']['root'],
                                       self.args['dataset']['file']['train_path'])
        dev_file_path = os.path.join(self.args['dataset']['file']['root'],
                                     self.args['dataset']['file']['dev_path'])

        train_timestamp_path = os.path.join(self.args['dataset']['timestamp']['root'],
                                            self.args['dataset']['timestamp']['train_path'])
        dev_timestamp_path = os.path.join(self.args['dataset']['timestamp']['root'],
                                          self.args['dataset']['timestamp']['dev_path'])

        train_audio_path = os.path.join(self.args['dataset']['audio']['root'],
                                        self.args['dataset']['audio']['train_path'])
        dev_audio_path = os.path.join(self.args['dataset']['audio']['root'],
                                      self.args['dataset']['audio']['dev_path'])

        train_observations = self.generate_observations_from_dataset(
            train_file_path, train_timestamp_path, train_audio_path)
        dev_observations = self.generate_observations_from_dataset(
            dev_file_path, dev_timestamp_path, dev_audio_path)
        
        return train_observations, dev_observations  

    def generate_observations_from_dataset(self, filepath, timestamp_path, audiopath):
        """
        Generates Observation objects for each audio in a directory.
        Each observation object is a list of (word, audio_embedding, w2v2_embeddings, NER)
        Args: the filesystem path to the conll dataset
        Returns: a list of Observations
        """
        observations = []

        file = open(filepath)
        lines = list(csv.reader(file, delimiter="\t"))

        for line in lines[1:]:  # a single audio
            file_id, transcript, labels = self.get_info_from_SLUE_line(line)
            audio_file = os.path.join(audiopath, file_id + ".ogg")
            timestamp_file = os.path.join(timestamp_path, file_id + ".csv")

            if os.path.isfile(audio_file) == False or os.path.isfile(timestamp_file) == False:
                continue
            
            embeddings = self.generate_embeddings(audio_file, timestamp_file)
            labels = self.convert_raw_label_to_list(transcript, labels)
            IOB_tags = self.IOB_tag_labels(labels)

            observation = self.observation_class(
                file_id, transcript, labels, IOB_tags, embeddings)
            observations.append(observation)
        return observations

    def generate_embeddings(self, audio_file, timestamps):
        audio, rate = sf.read(audio_file)
        audio = torch.tensor(audio, dtype=torch.float32).cuda()
        
        layer_index = self.args['model']['model_layer']  # from 1-24
        input_values = self.tokenizer(audio, return_tensors="pt").input_values
        self.model.eval()
        self.model.cuda()
        with torch.no_grad():
            output_values = self.model(input_values)
        layer_hidden_features = output_values[2][layer_index].detach().numpy().cpu()

        embd_list = []
        for word_ts in timestamps:
            start, end, word, type = word_ts
            if type == "words":
                start_frame = int(float(start) * 50) # SR=16000, Downsample=320
                end_frame = int(float(end) * 50)
                word_features = layer_hidden_features[:,
                                                  start_frame:end_frame+1, :]
                word_features = np.mean(word_features, axis=1)
                embd_list.append(word_features)
        return embd_list

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
            char_idx += len(word) + 1

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
            elif label == "CARDINAL" or label == "MONEY" or label == "ORDINAL" or label == "PERCENT" or label == "QUANTITY":
                IOB.append(prefix + "QUANT")
            elif label == "DATE" or label == "TIME":
                IOB.append(prefix + "WHEN")
            elif label == "ORG" or label == "NORP" or label == "PERSON" or label == "LAW":
                IOB.append(prefix + label)
            else:  # disgard rare labels
                IOB.append("O")

            prefix = "I-"
            last_label = label

        assert len(labels) == len(IOB)
        return IOB

    def get_info_from_SLUE_line(self, line):
        """
        tsv file is formatted as 
        id  raw_text normalized_text speaker_id split raw_ner normalized_ner
        """
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

    def tags(self):
        return ["PLACE", "QUANT", "WHEN", "ORG", "NORP", "PERSON", "LAW"]


class ObservationBatch:
    """
    Used for custom follate fn for observation iterator. 
    Flattens both embeddings and IOB tags so length of embeddings/tags is total word count in batch.
    """

    def __init__(self, data):
        """convert into list of lists of format:
            [[fileid1, transcript1, word_audio_vectors1, ...]
            [fileid2, ..]]"""

        self.data = [list(obs) for obs in data]
        file_ids, transcripts, labels, IOB_tags, embeddings = map(
            list, zip(*self.data))

        IOB_tags = [(self.integrize_labels(label)) for label in IOB_tags]
        self.IOB_tags = torch.tensor(np.vstack(IOB_tags))
        self.embeddings = torch.tensor(np.vstack(embeddings), dtype=torch.float32)
        assert self.IOB_tags.shape[0] == self.embeddings.shape[0]

    def integrize_labels(self, IOB_tag):
        all_IOB_labels = ["O", "B-PLACE", "I-PLACE", "B-QUANT", "I-QUANT", "B-WHEN", "I-WHEN",
                          "B-ORG", "I-ORG", "B-NORP", "I-NORP", "B-PERSON", "I-PERSON", "B-LAW", "I-LAW"]
        tags = []
        for word_tag in IOB_tag:
            tags.append(all_IOB_labels.index(word_tag))
        return tags


class ObservationIterator(Dataset):
    """ List Container for lists of Observations and labels for them.
    Used as the iterator for a PyTorch dataloader.
    """

    def __init__(self, observations):
        self.observations = observations

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        # consider put everything for loading into get item instead
        return self.observations[idx]
