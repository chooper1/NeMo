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


import os
import random
import json
import numpy as np
import librosa
import soundfile as sf

from nemo.collections.asr.parts.utils.speaker_utils import labels_to_rttmfile

#from scripts/speaker_tasks/filelist_to_manifest.py - move function?
def read_manifest(manifest):
    data = []
    with open(manifest, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

class LibriSpeechGenerator(object):
    """
    Librispeech Diarization Session Generator.

    Args:
        manifest_path (str): Manifest file with paths to librispeech audio files
        sr (int): sampling rate of the audio files
        num_speakers (int): number of unique speakers per diarization session
        session_length (int): length of each diarization session (seconds)
        output_dir (str): output directory
        output_filename (str): output filename for the wav and rttm files
        rng: Random number generator
    """
    def __init__(
        self,
        manifest_path=None,
        sr=16000,
        num_speakers=2,
        session_length=60,
        output_dir='output',
        output_filename='librispeech_diarization',
        rng=None,
    ):
        self._manifest_path = manifest_path
        self._sr = sr
        self._num_speakers = num_speakers
        self._session_length = session_length
        self._output_dir = output_dir
        self._output_filename = output_filename
        self._rng = random.Random() if rng is None else rng

        self._manifest = read_manifest(manifest_path)

    #Get/Set Methods
    def set_session_length(self, new_sl):
        self._session_length = new_sl

    def set_output_filename(self, new_fn):
        self._output_filename = new_fn

    def set_num_speakers(self, new_ns):
        self._num_speakers = new_ns

    #TODO add method to load all parameters from a config file (yaml)

    #randomly select speaker ids from loaded dict
    #TODO enforce exclusivity
    def get_speaker_ids(self):
        speaker_ids = []
        for s in range(0,self._num_speakers):
            file = self._manifest[random.randint(0, len(self._manifest)-1)]
            fn = file['audio_filepath'].split('/')[-1]
            speaker_id = fn.split('-')[0]
            speaker_ids.append(speaker_id)
        return speaker_ids

    #get a list of the samples for the two specified speakers
    #TODO clean up dict usage (currently using librispeech id as index)
    def get_speaker_samples(self, speaker_ids):
        speaker_lists = {}
        for i in range(0,self._num_speakers):
            spid = speaker_ids[i]
            speaker_lists[str(spid)] = []

        for file in self._manifest:
            fn = file['audio_filepath'].split('/')[-1]
            new_speaker_id = fn.split('-')[0]
            for spid in speaker_ids:
                if spid == new_speaker_id:
                    speaker_lists[str(spid)].append(file)

        return speaker_lists

    #load a sample for the selected speaker id
    def load_speaker_sample(self, speaker_lists, speaker_ids, speaker_turn):
        speaker_id = speaker_ids[speaker_turn]
        file_id = random.randint(0,len(speaker_lists[str(speaker_id)])-1)
        file = speaker_lists[str(speaker_id)][file_id]
        return file

    #add new entry to dict (to write to output manifest file)
    def create_new_rttm_entry(self, new_file, start, speaker_id):
        end = start + new_file['duration']
        return str(start) + ' ' + str(end) + ' ' + str(speaker_id)

    #generate diarization session
    def generate_session(self):
        speaker_ids = self.get_speaker_ids() #randomly select 2 speaker ids
        speaker_lists = self.get_speaker_samples(speaker_ids) #get list of samples per speaker

        speaker_turn = 0 #assume alternating between speakers 1 & 2
        running_length = 0

        wavpath = os.path.join(self._output_dir, self._output_filename + '.wav')
        array = np.zeros(self._session_length*self._sr)
        manifest_list = []

        while (running_length < self._session_length):
            file = self.load_speaker_sample(speaker_lists, speaker_ids, speaker_turn)
            filepath = file['audio_filepath']
            duration = file['duration']
            audio_file, sr = librosa.load(filepath, sr=self._sr)

            # Reintroduce once frame-level word alignments are available?
            # if (running_length + duration) > self._session_length:
            #     duration = self._session_length - running_length

            start = int(running_length*self._sr)
            length = int(duration*self._sr)

            # Remove once frame-level word alignments are available?
            print(start)
            print(length)
            print(array.shape)
            print(audio_file.shape)
            print(self._session_length*self._sr)
            if (start+length > self._session_length*self._sr):
                print('pad')
                np.pad(array, pad_width=(0, start+length-self._session_length*self._sr), mode='constant')
            array[start:start+length] = audio_file[:length]

            new_entry = self.create_new_rttm_entry(file, running_length, speaker_ids[speaker_turn])
            manifest_list.append(new_entry)

            #pick new speaker
            prev_speaker_turn = speaker_turn
            speaker_turn = random.randint(0, self._num_speakers-1)
            while (speaker_turn == prev_speaker_turn):
                speaker_turn = random.randint(0, self._num_speakers-1)

            running_length += duration

        sf.write(wavpath, array, self._sr)
        labels_to_rttmfile(manifest_list, self._output_filename, self._output_dir)
