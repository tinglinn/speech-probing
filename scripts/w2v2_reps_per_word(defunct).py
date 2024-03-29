import torch
import librosa
import h5py
import csv
import os
import numpy as np
from transformers.utils.dummy_pt_objects import Wav2Vec2PreTrainedModel
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Tokenizer
from argparse import ArgumentParser

# import pretrained model and tokenizer
config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-lv60", output_hidden_states=True)
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-lv60", config=config)
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-lv60")

argp = ArgumentParser()
argp.add_argument('input_path')
argp.add_argument('output_path')
argp.add_argument('time_stamp_path')
argp.add_argument('pool_type')
args = argp.parse_args()

LAYER_COUNT = 24
FEATURE_COUNT = 1024
MODEL_DOWNSAMPLING = 320
SAMPLE_RATE = 16000
model.eval()

with h5py.File(args.output_path, 'w') as fout:
    for index, audio_file in enumerate(os.scandir(args.input_path)):
        waveform, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
        
        # use file_id to open time stamps (assuming csv file)
        file_id = os.path.basename(audio_file).split(".")[0]
        time_stamps = os.path.join(args.time_stamp_path, file_id + ".csv")
        
        # get w2v2 features
        input_values = tokenizer(waveform, return_tensors = "pt").input_values
        with torch.no_grad():
            output_values = model(input_values)
        
        hidden_features = output_values[2][1:]

        # get features per word
        features_list = []
        word_count = 0
        with open(time_stamps) as fd:
            rd = csv.reader(fd, delimiter=",")
            for row in rd: 
                text_type = row# one word
                word_count += 1
                start_frame = int(float(row[0]) * (SAMPLE_RATE / MODEL_DOWNSAMPLING))
                end_frame = int(float(row[1]) * (SAMPLE_RATE / MODEL_DOWNSAMPLING))
                word_features = np.vstack([np.array(x)[:,start_frame:end_frame,:] for x in hidden_features])
                if args.pool_type == "mean":
                    word_features = np.mean(word_features, axis=1)
                elif args.pool_type == "max":
                    word_features = np.max(np.array(word_features), axis=1)
                elif args.pool_type == "first":
                    word_features = word_features[:,0,:]
                features_list.append(word_features)

        
        key = file_id
        try:
            # should dset key map to a list of features or to a 3d np array?
            dset = fout.create_dataset(key, (word_count, LAYER_COUNT, FEATURE_COUNT))
        except RuntimeError:
            dset = fout[key]

        dset[:,:,:] = np.stack([np.array(x) for x in features_list])
