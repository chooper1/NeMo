# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import random
import shutil
import numpy as np

random.seed(42)

"""
This script parses a CMI file to extract statistics from the AMI dataset.
"""

def read_cmi_files(directory_path):
    onlyfiles = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    data = {}
    for file in onlyfiles:
        data[str(file)] = []
        with open(os.path.join(directory_path, file), 'r', encoding='utf-8') as f:
            for line in f:
                data[str(file)].append(line.strip('\n').split(' '))
    return data


def main():
    input_directory = args.input_directory
    sentence_break_time = args.sentence_threshold
    sample_rate = args.sample_rate
    bin_size = args.bin_size
    list = read_cmi_files(input_directory)

    #collect statistics across all AMI files
    full_silence_percent = []
    full_overlap_percent = []
    full_silence_lengths = []
    full_overlap_lengths = []
    full_dominance_var = []
    full_dominance_stddev = []
    total_sentence_lengths = {}
    total_num_speakers = {}

    yes_turn = 0
    no_turn = 0

    for key in list:
        meeting = list[key]

        current_sentence_lengths = {}
        prev_time_per_speaker = {} #use for determining end of sentences
        dominance_per_speaker = {} #track per-speaker speaking time
        prev_sp = None

        #used in second loop to get overlap / silence
        largest_end_time = 0

        for line in meeting:
            sp = line[1]
            start = float(line[2])
            dur = float(line[3])
            end = start+dur

            if(end > largest_end_time):
                largest_end_time = end

            #first time this speaker has spoken
            if str(sp) not in prev_time_per_speaker:
                prev_time_per_speaker[str(sp)] = end
                current_sentence_lengths[str(sp)] = 1
                dominance_per_speaker[str(sp)] = 0

            #break sentence if it has been one second since this speaker spoke
            elif start - prev_time_per_speaker[str(sp)] > sentence_break_time:
                #record sentence length, whether there was a speaker turn
                if str(current_sentence_lengths[str(sp)]) not in total_sentence_lengths:
                    total_sentence_lengths[str(current_sentence_lengths[str(sp)])] = 0
                total_sentence_lengths[str(current_sentence_lengths[str(sp)])] += 1
                current_sentence_lengths[str(sp)] = 1
                if prev_sp == sp:
                    no_turn += 1
                else:
                    yes_turn += 1

            #else continue sentence
            else:
                current_sentence_lengths[str(sp)] += 1

            prev_time_per_speaker[str(sp)] = end
            dominance_per_speaker[str(sp)] += end - start
            prev_sp = sp

        #get speaking time, overlap, silence
        timeline = np.zeros(int(largest_end_time*100))
        for line in meeting:
            sp = line[1]
            start = int(float(line[2])*100)
            dur = int(float(line[3])*100)
            end = start+dur
            timeline[start:end] += 1

        speaking_time = np.sum(timeline > 0)
        silence_time = len(timeline) - speaking_time
        overlap_time = np.sum(timeline > 1)

        overlap_lengths = []
        overlap_count = 0
        silence_lengths = []
        silence_count = 0
        for i in range(0, len(timeline)):

            if timeline[i] == 0:
                silence_count += 1
            else:
                if silence_count > 0:
                    silence_lengths.append(silence_count)
                silence_count = 0

            if timeline[i] > 1:
                overlap_count += 1
            else:
                if overlap_count > 0:
                    silence_lengths.append(overlap_count)
                overlap_count = 0

        silence_percent = silence_time / len(timeline)
        overlap_percent = overlap_time / speaking_time

        full_silence_percent.append(silence_percent)
        full_overlap_percent.append(overlap_percent)

        #get speaker dominance variance
        dominance = []
        total_dominance = 0
        for k in dominance_per_speaker:
            total_dominance += dominance_per_speaker[k]
        for k in dominance_per_speaker:
            dominance_per_speaker[k] = dominance_per_speaker[k] / total_dominance
            dominance.append(dominance_per_speaker[k])

        dvar = np.var(dominance)
        full_dominance_var.append(dvar)
        full_dominance_stddev.append(np.sqrt(dvar))

        num_speakers = len(prev_time_per_speaker)
        if str(num_speakers) not in total_num_speakers:
            total_num_speakers[str(num_speakers)] = 1
        else:
            total_num_speakers[str(num_speakers)] += 1

        full_silence_lengths += silence_lengths
        full_overlap_lengths += overlap_lengths

    silence_binned = {}
    overlap_binned = {}
    print(full_silence_lengths)
    print(full_overlap_lengths)

    for i in range(0,len(full_silence_lengths)):
        length = full_silence_lengths[i]*1.0 / sample_rate
        len_rounded = round(length * (1.0/bin_size)) / (1.0/bin_size)

        if str(len_rounded) not in silence_binned:
            silence_binned[str(len_rounded)] = 0
        silence_binned[str(len_rounded)] += 1

    for i in range(0,len(full_overlap_lengths)):
        length = full_overlap_lengths[i]*1.0 / sample_rate
        len_rounded = round(length * (1.0/bin_size)) / (1.0/bin_size)

        if str(len_rounded) not in overlap_binned:
            overlap_binned[str(len_rounded)] = 0
        overlap_binned[str(len_rounded)] += 1


    #replace with logging?
    print('full_silence_percent: ', np.mean(full_silence_percent))
    print('full_overlap_percent: ', np.mean(full_overlap_percent))
    print('full_dominance_var: ', np.mean(full_dominance_var))
    print('full_dominance_stddev: ', np.mean(full_dominance_stddev))
    print('turn_prob: ', float(yes_turn) / (float(yes_turn)+float(no_turn)))
    print('full_num_speakers: ', total_num_speakers)
    print('full_total_sentence_lengths: ', total_sentence_lengths)

    print('silence_binned: ', silence_binned)
    print('overlap_binned: ', overlap_binned)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMI CMI file parser")
    parser.add_argument("--input_directory", help="Path to input CMI file", type=str, required=True) #change eventually to loop over all files in directory
    parser.add_argument("--sentence_threshold", help="Sentence Threshold", type=float, default=1.0)
    parser.add_argument("--sample_rate", help="Sampling Rate", type=int, default=16000)
    parser.add_argument("--bin_size", help="Bin Size", type=float, default=0.05)
    args = parser.parse_args()

    main()
