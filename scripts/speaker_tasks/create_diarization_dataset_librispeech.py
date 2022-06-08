# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from pydub import AudioSegment
from filelist_to_manifest import read_manifest #TODO add support for multiple input manifest files?
# from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.speaker_utils import labels_to_rttmfile

random.seed(42)

"""
This script creates a synthetic diarization dataset using the LibriSpeech dataset.
"""

# using modified version that appends to the current rttm file
# previous version: from nemo.collections.asr.parts.utils.speaker_utils import labels_to_rttmfile
# def labels_to_rttmfile(labels, uniq_id, filepath):
#     """
#     write rttm file with filepath for wav file uniq_id with time_stamps in labels
#     """
#     with open(filepath, 'a') as f:
#         for line in labels:
#             line = line.strip()
#             start, end, speaker = line.split()
#             duration = float(end) - float(start)
#             start = float(start)
#             log = 'SPEAKER {} 1   {:.3f}   {:.3f} <NA> <NA> {} <NA> <NA>\n'.format(uniq_id, start, duration, speaker)
#             f.write(log)
#
#     return filepath

#randomly select 2 speaker ids from loaded dict
#TODO make parameterizable
#TODO enforce exclusivity
def get_speaker_ids(list):
    file1 = list[random.randint(0, len(list)-1)]
    file2 = list[random.randint(0, len(list)-1)]

    fn1 = file1['audio_filepath'].split('/')[-1]
    fn2 = file2['audio_filepath'].split('/')[-1]

    speaker_id1 = fn1.split('-')[0]
    speaker_id2 = fn2.split('-')[0]

    return [speaker_id1,speaker_id2]

#get a list of the samples for the two specified speakers
#TODO enforce exclusion within one clip (avoid repetition)?
#TODO replace with more efficient sampling method?
def get_speaker_samples(file_list, speaker_ids):
    speaker_lists = {'sp1': [], 'sp2': []}
    for file in file_list:
        fn = file['audio_filepath'].split('/')[-1]
        spid = fn.split('-')[0]
        if spid == speaker_ids[0]:
            speaker_lists['sp1'].append(file)
        elif spid == speaker_ids[1]:
            speaker_lists['sp2'].append(file)
    return speaker_lists

#load a sample for the selected speaker id
def load_speaker_sample(speaker_lists, speaker_turn):
    if (speaker_turn == 0):
        speaker_id = 'sp1'
    elif (speaker_turn == 1):
        speaker_id = 'sp2'
    file_id = random.randint(0,len(speaker_lists[speaker_id])-1)
    file = speaker_lists[speaker_id][file_id]
    return file

#add new entry to dict (to write to output manifest file)
def create_new_entry(new_file, start, speaker_id):
    end = start + new_file['duration']
    return str(start) + ' ' + str(end) + ' ' + str(speaker_id)

def main(
    input_manifest_filepath, output_dir, output_filename = 'librispeech_diarization', session_length = 20, num_sessions=1
):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    #load librispeech manifest file
    input_file = read_manifest(input_manifest_filepath)
    manifest_list = []
    rttm_path = os.path.join(output_dir, output_filename + '.rttm')

    for session in range(0,num_sessions):
        #get speaker ids for a given diarization session
        speaker_ids = get_speaker_ids(input_file) #randomly select 2 speaker ids
        speaker_lists = get_speaker_samples(input_file, speaker_ids) #get list of samples per speaker
        session_filename = output_filename + '_{}'.format(session)

        speaker_turn = 0 #assume alternating between speakers 1 & 2
        running_length = 0

        wavpath = os.path.join(output_dir, session_filename + '.wav')
        # out_file = AudioSegment.silent(duration=session_length*1000).set_frame_rate(16000)
        array = np.zeros(session_length*16000)
        # out_file = AudioSegment(zeros,16000)
        # out_file.pad(session_length*16000)

        while (running_length < session_length):
            file = load_speaker_sample(speaker_lists, speaker_turn)
            filepath = file['audio_filepath']
            duration = file['duration']
            if (running_length+duration) > session_length:
                duration = session_length - running_length

            audio_file = AudioSegment.from_wav(filepath).set_frame_rate(16000)
            # audio_file = AudioSegment.from_file(filepath, target_sr=16000)

            start = int(running_length*16000)
            length = int(duration*16000)
            # out_file._samples[start:start+length] = audio_file._samples[:length]
            array[start:start+length] = audio_file[:length]

            # silent_duration = 0.25 #0.25 blank seconds
            # blank = AudioSegment.silent(duration=silent_duration*1000)
            # out_file += blank

            #TODO fixed size dict before loop?
            new_entry = create_new_entry(file, running_length, speaker_ids[speaker_turn])
            manifest_list.append(new_entry)

            speaker_turn = (speaker_turn + 1) % 2
            running_length += duration

        # wav_out.close()
        out_file = AudioSegment.from_numpy_array(array)
        out_file.export(wavpath, format="wav")
        # labels_to_rttmfile(manifest_list, session_filename, rttm_path)
        labels_to_rttmfile(manifest_list, session_filename, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest_filepath", help="path to input manifest file", type=str, required=True)
    parser.add_argument("--output_dir", help="path to output directory", type=str, required=True)
    args = parser.parse_args()

    main(
        args.input_manifest_filepath,
        args.output_dir,
    )
