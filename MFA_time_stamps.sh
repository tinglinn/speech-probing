"""
Processing audio files using the Montreal Forced Aligner (MFA) to obtain
time stamps.

Adapted from: https://gist.github.com/NTT123/12264d15afad861cb897f7a20a01762e

"""

# MFA: https://montreal-forced-aligner.readthedocs.io/en/latest/aligning.html"

root_dir=${1:-/tmp/mfa}
mkdir -p $root_dir
cd $root_dir
source $root_dir/miniconda3/bin/activate aligner

# Install mfa, download kaldi
pip install montreal-forced-aligner # install requirements
pip install git+https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner.git # install latest updates

mfa thirdparty download

# create transcript files from metadata.csv
lines = open('./FILEPATH', 'r').readlines()
from tqdm.auto import tqdm
for line in tqdm(lines):
  fn, _, transcript = line.strip().split('|')
  ident = fn
  open(f'./wav/{ident}.txt', 'w').write(transcript)


# download a pretrained english acoustic model, and english lexicon
!wget -q --show-progress https://github.com/MontrealCorpusTools/mfa-models/raw/master/acoustic/english.zip
!wget -q --show-progress http://www.openslr.org/resources/11/librispeech-lexicon.txt


# align phonemes and speech
!source {INSTALL_DIR}/miniconda3/bin/activate aligner; \
mfa align -t ./temp -c -j 4 ./wav librispeech-lexicon.txt ./english.zip ./'OUTPUT PATH'