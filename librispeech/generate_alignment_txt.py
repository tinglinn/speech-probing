"""
Takes the raw output csv files from the Montreal Forced Aligner and writes all 
information to a single txt file for each dataset split. 
"""

import csv
from ntpath import join
import os
import argparse
from typing import Concatenate

def generate_alignment_list(file_directory):
    alignment_list = []
    for speaker in os.listdir(file_directory):
        speaker_dir = os.path.join(file_directory, speaker)
        alignment_files = os.listdir(speaker_dir)
        
        for file in alignment_files:  # one audio -> one line in output
            this_line = ""
            id = os.path.basename(file).split(".")
            transcripts = []
            timestamps = []
            with open(file) as f:
                csv_file = csv.reader(f, delimiter=",")
                for row in csv_file:  # each row: Begin,End,Label,Type,Speaker
                    begin, end, label, type, _ = row
                    if type == "words":
                        transcripts.append(label)
                        timestamps.append(begin)
                        timestamps.append(end)
            print(transcripts)
            print(timestamps)
            transcripts = ",".join(transcripts)
            timestamps = ",".join(timestamps)
            this_line = Concatenate(id, " ", transcripts, " ", timestamps)
            print(this_line)
            return
            alignment_list.append(this_line)
    return alignment_list

def write_alignments(file_directory, output_file):
    all_lines = generate_alignment_list(file_directory)
    with open(output_file, 'w') as f:
        for line in all_lines:
            f.write(f"{line}\n")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file_directory', type=str,
                        help='Path to alignment csv files')
    parser.add_argument('output_file', type=str,
                        help='Path to output txt files')
    args = parser.parse_args()
    write_alignments(args.file_directory, args.output_file)
