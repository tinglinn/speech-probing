# speech-probing

This repository is a code base for probing audio transformers, specifically Wav2vec 2.0, for linguistic understanding. We use two probing tasks: 1) POS tagging, and 2) named entity recognition. 

# set-up

Experiments run with this repository are specified via yaml files that completely describe the experiment. There are two types of yaml files: one for monolingual experiments and one for cross-lingual experiments. 

# monolingual probing

To run an experiment, enter the command ```python run_experiment.py config.yaml```. 

# cross-lingual probing

To run a cross-lingual experiment, enter the command ```python run_cl_experiment.py config_cl.yaml```. It trains a linear probe on one language and evaluates on another. This measures the amount of cross-lingual transfer present in the multilingual audio transformer.
