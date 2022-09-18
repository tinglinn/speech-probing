"""
A script for obtaining stanza POS (upos/xpos) annotations in batches and 
writing it into a txt file. It takes in a txt file with file_id, transcript,
and time stamps for each audio. For more accurate results, we pass through
CoreNLP true-casing before calling stanza.

Adapted from Ethan Chi.
"""

import json
import sys
from collections import namedtuple

import h5py
from tqdm.auto import tqdm as tqdm
import stanza
from stanza.server import CoreNLPClient

Observation = namedtuple('Observation', ('id', 'words', 'times'))

fold = sys.argv[1]

with open(f'{fold}.alignment.txt') as alignments:
  lines = [line.strip() for line in alignments]
  lines = [line for line in lines if line]

observations = []
print(len(lines))
for line in lines:
  id, words, times = line.split('\t')
  words = [w for w in words.split(',')]
  times = [float(time) for time in times.split(',')]
  times = [(times[i], times[i+1]) for i in range(0, len(times), 2)]
  words, times = zip(*[(words[i], times[i]) for i in range(len(words))])
  if len(words) > 200:
      continue
  observations.append(Observation(id, words, times))

BATCH_SIZE = 200
batches = ['\n'.join(' '.join(observation.words) for observation in observations[i:i+BATCH_SIZE])
           for i in range(0, len(observations), BATCH_SIZE)]
print(batches[0])

cased_sentences = []

with CoreNLPClient(
        annotators=['tokenize', 'ssplit', 'truecase'],
        properties={
            'tokenize.language': 'whitespace',
            'truecase.overwriteText': True,
            'ssplit.eolonly': True,
        },
        timeout=30000,
        memory='16G',
        be_quiet=True) as client:
    for batch in tqdm(batches):
        ann = client.annotate(batch)
        for sentence in ann.sentence:
            new_words = [token.word for token in sentence.token]
            cased_sentences.append(new_words)

# {'tokenize': 'ewt', 'pos': 'ewt', 'ner': 'CoNLL03'}
nlp = stanza.Pipeline('en', processors='tokenize,pos,ner', ner_model_path='/sailhome/ethanchi/stanza_resources/en/ner/conll03.pt',
                      tokenize_pretokenized=True, verbose=True, logging_level='DEBUG')
out = nlp('\n'.join(' '.join(words) for words in cased_sentences))
ner = [[token.ner for token in sentence.tokens] for sentence in out.sentences]
upos = [[token.words[0].upos for token in sentence.tokens]
        for sentence in out.sentences]
xpos = [[token.words[0].xpos for token in sentence.tokens]
        for sentence in out.sentences]


cased_observations = [(observation.id, cased_words, observation.times, {
                       'ner': n, 'upos': u, 'xpos': x}) for observation, cased_words, n, u, x in zip(observations, cased_sentences, ner, upos, xpos)]

with open(f'{fold}.txt', 'w') as f:
    json.dump(cased_observations, f, indent=2)
