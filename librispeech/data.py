from collections import defaultdict, Counter, namedtuple
import os
import json
import soundfile as sf
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers.utils.dummy_pt_objects import Wav2Vec2PreTrainedModel
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Tokenizer

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-lv60", output_hidden_states=True)
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-lv60", config=config)
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-lv60")

        self.batch_size = args['dataset']['batch_size']
        self.observation_class = self.get_observation_class(
            self.args['dataset']['observation_fieldnames'])
        self.train_obs, self.dev_obs = self.read_from_disk()  # commented out self.test_obs
        self.train_dataset = ObservationIterator(self.train_obs)
        self.dev_dataset = ObservationIterator(self.dev_obs)
        print("Length of train dataset", len(self.train_dataset))
        print("Length of dev dataset", len(self.dev_dataset))
        

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
        task = self.args["task"]

        train_audio_path = os.path.join(self.args['dataset']['audio']['root'],
            self.args['dataset']['audio']['train_path'])
        dev_audio_path = os.path.join(self.args['dataset']['audio']['root'],
            self.args['dataset']['audio']['dev_path'])

        train_file_path = os.path.join(self.args['dataset']['file']['root'],
                                       self.args['dataset']['file']['train_path'])
        dev_file_path = os.path.join(self.args['dataset']['file']['root'],
                                     self.args['dataset']['file']['dev_path'])

        train_observations = self.generate_observations_from_dataset(
            train_file_path, train_audio_path, task)
        dev_observations = self.generate_observations_from_dataset(
            dev_file_path, dev_audio_path, task)
         
        return train_observations, dev_observations 

    def generate_observations_from_dataset(self, filepath, audiopath, task):
        """
        Generates Observation objects for each audio in a directory.
        Each observation object is a list of (word, audio_embedding, w2v2_embeddings, NER)
        Args: the filesystem path to the conll dataset
        Returns: a list of Observations
        """
        observations = []

        file = open(filepath)
        json_lines = json.load(file)

        for item in json_lines:  # a single audio
            file_id, transcript, timestamps, annotations = item
            audio = self.get_audio_from_id(file_id, audiopath)
            if task == "upos":
                labels = annotations[task]
            if task == "ner":
                labels = self.generate_IOB(annotations[task])
            # embeddings = [None for x in range(len(transcript))]
            embeddings = self.generate_embeddings(audio, timestamps)
            observation = self.observation_class(file_id, transcript, labels, embeddings)
            observations.append(observation)
        return observations
    
    def generate_IOB(labels):
        # change from BIOES encoding to IOB encoding
        return [label.replace('S-', 'B-').replace('E-', 'I-') for label in labels]

    def get_audio_from_id(self, file_id, audiopath):
        speaker_id = file_id.split("-")[0]
        filepath = os.path.join(audiopath, speaker_id, file_id + ".wav")
        audio, rate = sf.read(filepath)
        audio = torch.tensor(audio, dtype=torch.float32).cuda()
        return audio

    def generate_embeddings(self, audio, timestamps):
        layer_index = self.args['model']['model_layer']  # from 1-24
        input_values = self.tokenizer(audio, return_tensors="pt").input_values
        self.model.eval()
        self.model.cuda()
        with torch.no_grad():
            output_values = self.model(input_values)
        layer_hidden_features = output_values[2][layer_index].detach().numpy().cpu()  
    
        embd_list = []
        for word_ts in timestamps:
            start, end = word_ts
            start_frame = int(float(start) * 50)  # SR=16000, Downsample=320
            end_frame = int(float(end) * 50)
            word_features = layer_hidden_features[:, start_frame:end_frame+1,:]
            word_features = np.mean(word_features, axis=1)
            embd_list.append(word_features)
        return embd_list

    def generate_random_embeddings(self, word_count):
        """
        For control task. Generates random embedding of target size. 
        """
        embd_list = []
        for i in range(word_count):
            random_embd = torch.rand(1, 1024, dtype=torch.float32)
            embd_list.append(random_embd)
        return embd_list
    
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
        batch = ObservationBatch(batch)
        return batch.embeddings, batch.labels


class ObservationBatch:
    """
    Used for custom follate fn for observation iterator. 
    Flattens both embeddings and IOB tags so length of embeddings/tags is total word count in batch.
    """

    def __init__(self, data):
        """convert into list of lists of format:
            [[fileid1, transcript1, labels1, ...]
            [fileid2, ..]]"""

        self.data = [list(obs) for obs in data]
        file_ids, transcripts, labels, embeddings = map(
            list, zip(*self.data))

        labels = [(self.integrize_labels(label)) for label in labels]
        self.labels = torch.tensor(np.vstack(labels))
        self.embeddings = torch.tensor(np.vstack(embeddings), dtype=torch.float32)
        assert self.labels.shape[0] == self.embeddings.shape[0]

    def integrize_labels(self, label):
        all_labels = ["ADJ", "ADP", "ADV", "AUX", "CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]
        labels = []
        for word_label in label:
            labels.append(all_labels.index(word_label))
        return labels


class ObservationIterator(Dataset):
    """ List Container for lists of Observations and labels for them.
    Used as the iterator for a PyTorch dataloader.
    """

    def __init__(self, observations):
        self.observations = observations

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx]
