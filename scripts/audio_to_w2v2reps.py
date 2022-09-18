"""
Takes audio file and saves wav2vec2(L-lv-60k) features for all 24 layers to disk.

Adapted from: https://github.com/john-hewitt/structural-probes/blob/master/scripts/convert_raw_to_bert.py

"""

import torch
import librosa
import h5py
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
args = argp.parse_args()

LAYER_COUNT = 24
FEATURE_COUNT = 1024

model.eval()

with h5py.File(args.output_path, 'w') as fout:
    # instead of using index, maybe use file id? unsure
    for index, audio_file in enumerate(open(args.input_path)):
        waveform, sr = librosa.load(audio_file, sr=16000)
        input_values = tokenizer(waveform, return_tensors = "pt").input_values
        with torch.no_grad():
            output_values = model(input_values)
        hidden_features = output_values[2][1:]
        
        file_id = os.file.basename(audio_file).split(".")[0]
        try:
            # should dset key map to a list of features or to a 4d np array?
            dset = fout.create_dataset(key, (word_count, LAYER_COUNT, hidden_features[-1].shape[1], FEATURE_COUNT))
        except RuntimeError:
            dset = fout[key]
            
        dset[:,:,:] = np.vstack([np.array(x) for x in hidden_features])



    
        
