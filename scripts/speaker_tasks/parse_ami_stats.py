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
    list = read_cmi_files(input_directory)

    sentence_break_time = 1.0 #1 second

    for key,meeting in list:
        silence_time = 0
        speaking_time = 0
        overlap_time = 0
        sentence_lengths = {}

        prev_sp = None
        sentence_length = 0
        current_start = 0
        prev_end = 0
        prev_time_per_speaker = {}

        largest_end_time = 0

        for line in meeting:
            sp = line[1]
            start = float(line[2])
            dur = float(line[3])
            end = start+dur

            if(end > largest_end_time):
                largest_end_time = end

            sentence_length += 1

            #deal with overlap not breaking up sentence
            if prev_sp != sp and start - prev_time_per_speaker[str(sp)] > sentence_break_time:
                if not sentence_lengths.has_key(sentence_length):
                    sentence_lengths[str(sentence_length)] = 0
                sentence_lengths[str(sentence_length)] += 1
                sentence_length = 0
                silence_time += start - prev_end #what about overlap?

            prev_time_per_speaker[str(sp)] = prev_end = end

        for line in meeting:







    #0 - file id, 1 - speaker id, 2 - start time, 3 - duration, 4 - word
    #assume break greater than one second is end of a sentence

    #desired stats:
    #   -sentence_length_params=[2.81, 0.1]
    #   -dominance_var=0.1
    #   -turn_prob=0.9
    #   -overlap_prob=0.3
    #   -mean_overlap=0.08
    #   -dist'n of overlap
    #   -mean_silence=0.08
    #   -dist'n of silence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMI CMI file parser")
    parser.add_argument("--input_directory", help="Path to input CMI file", type=str, required=True) #change eventually to loop over all files in directory
    args = parser.parse_args()

    main()
