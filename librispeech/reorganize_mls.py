"""
Reorganizes a Multilingual LibriSpeech dataset into a format required by
the Montreal Forced Aligner. Converts all flac files to wav files and 
writes transcripts in lab format.

Adapted from: https://github.com/MontrealCorpusTools/MFA-reorganization-scripts/blob/main/Librispeech/reorganize_librispeech.py
"""

import argparse
import os
import subprocess


sox_path = "/nlp/scr/linting/miniconda3/bin/sox"


def get_transcripts(path):
    transcripts = {}
    with open(path) as f:
        for line in f:
            file, transcript = line.strip().split('\t', maxsplit=1)
            transcripts[file] = transcript
    return transcripts


def convert_flac(path, out_path):
    subprocess.call([sox_path, path, out_path])


def reorganize_dir(librispeech_directory, output_directory):
    audio_directory = os.path.join(librispeech_directory, "audio")
    transcript_path = os.path.join(librispeech_directory, "transcripts.txt")
    transcripts = get_transcripts(transcript_path)
    
    for k, v in transcripts.items():
        print(k)
        speaker = k.split("_")[0]
        out_speaker_dir = os.path.join(output_directory, speaker)
        os.makedirs(out_speaker_dir, exist_ok=True)
        ts_path = os.path.join(out_speaker_dir, k + '.lab')
        with open(ts_path, 'w', encoding='utf8') as f:
            f.write(v)

    for speaker in os.listdir(audio_directory):
        speaker_dir = os.path.join(audio_directory, speaker)
        out_speaker_dir = os.path.join(output_directory, speaker)
        #os.makedirs(out_speaker_dir, exist_ok=True)
        for chapter in os.listdir(speaker_dir):
            chapter_dir = os.path.join(speaker_dir, chapter)
            sound_files = os.listdir(chapter_dir)
            for sf in sound_files:
                if not sf.endswith('.flac'):
                    continue
                sf_path = os.path.join(chapter_dir, sf)
                sf_out_path = os.path.join(
                    out_speaker_dir, sf.replace('.flac', '.wav'))
                convert_flac(sf_path, sf_out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('librispeech_directory', type=str,
                        help='Path to LibriSpeech')
    parser.add_argument('output_directory', type=str,
                        help='Path to output directory')
    args = parser.parse_args()
    reorganize_dir(args.librispeech_directory, args.output_directory)
