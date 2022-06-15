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
    print(list)

    sentence_break_time = 1.0 #1 second

    for key in list:
        meeting = list[key]
        prev_sp = None
        sentence_length = 0
        current_start = 0
        prev_end = 0

        total_sentence_lengths = {}
        current_sentence_lengths = {}
        prev_time_per_speaker = {}

        largest_end_time = 0

        for line in meeting:
            sp = line[1]
            start = float(line[2])
            dur = float(line[3])
            end = start+dur

            if(end > largest_end_time):
                largest_end_time = end

            if str(sp) not in prev_time_per_speaker:
                prev_time_per_speaker[str(sp)] = end
                current_sentence_lengths[str(sp)] = 1
            elif start - prev_time_per_speaker[str(sp)] > sentence_break_time:
                #break sentence
                if not total_sentence_lengths.has_key(current_sentence_lengths[str(sp)]):
                    total_sentence_lengths[str(current_sentence_lengths[str(sp)])] = 0
                total_sentence_lengths[str(current_sentence_lengths[str(sp)])] += 1
                current_sentence_lengths[str(sp)] = 0
            else:
                #continue sentence
                current_sentence_lengths[str(sp)] += 1

            prev_time_per_speaker[str(sp)] = end

        timeline = np.zeros(int(largest_end_time*100))

        for line in meeting:
            sp = line[1]
            start = int(float(line[2])*100)
            dur = int(float(line[3])*100)
            end = int((start+dur)*100)
            timeline[start:end] += 1

        speaking_time = np.sum(timeline > 0)
        silence_time = len(timeline) - speaking_time
        overlap_time = np.sum(timeline > 1)

        silence_percent = silence_time / len(timeline)
        overlap_percent = overlap_time / speaking_time

        print('speaking_time: ', speaking_time)
        print('silence_time: ', silence_time)
        print('overlap_time: ', overlap_time)

        print('silence_percent: ', silence_percent)
        print('overlap_percent: ', overlap_percent)

        print('total_sentence_lengths: ', total_sentence_lengths)

        #per speaker time
        break










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
