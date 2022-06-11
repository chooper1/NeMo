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
        min_silence (float): minimum silence between (non-overlapping) speakers
        max_silence (float): maximum silence between (non-overlapping) speakers
        overlap_frequency (float): frequency of overlapping speech
        min_overlap (float): minimum percentage of overlap (as a percentage of the first clip)
        max_overlap (float): maximum percentage of overlap (as a percentage of the first clip)
    """
    def __init__(
        self,
        manifest_path=None,
        sr=16000,
        num_speakers=2,
        session_length=60,
        output_dir='output',
        output_filename='librispeech_diarization',
        sentence_length_params = [2.81,0.1], #from https://www.researchgate.net/publication/318396023_How_will_text_size_influence_the_length_of_its_linguistic_constituents, p.209
        turn_prob = 0.1,
    ):
        self._manifest_path = manifest_path
        self._sr = sr
        self._num_speakers = num_speakers
        self._session_length = session_length
        self._output_dir = output_dir
        self._output_filename = output_filename

        self._sentence_length_params = sentence_length_params
        self._turn_prob = turn_prob

        #overlap/silence

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
    def get_speaker_ids(self):
        speaker_ids = []
        s = 0
        while (s < self._num_speakers):
            file = self._manifest[random.randint(0, len(self._manifest)-1)]
            fn = file['audio_filepath'].split('/')[-1]
            speaker_id = fn.split('-')[0]
            if (speaker_id not in speaker_ids): #enforce exclusivity
                speaker_ids.append(speaker_id)
                s += 1
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

    #sample from speakers
    #TODO account for speaker dominance
    def get_speaker(self, prev_speaker):
        if (prev_speaker == None):
            speaker_turn = random.randint(0,self._num_speakers-1)
        else:
            speaker_turn = random.randint(0,self._num_speakers-1)
            while (speaker_turn == prev_speaker and random.uniform(0, 1) > self._turn_prob):
                speaker_turn = random.randint(0, self._num_speakers-1)
        return speaker_turn


    #generate diarization session
    def generate_session(self, num_sessions=1):
        for i in range(0,num_sessions):
            filename = self._output_filename + f"_{i}"

            speaker_ids = self.get_speaker_ids() #randomly select 2 speaker ids
            speaker_lists = self.get_speaker_samples(speaker_ids) #get list of samples per speaker

            speaker_turn = 0 #assume alternating between speakers 1 & 2
            running_length = 0
            previous_duration = 0 #for overlap

            wavpath = os.path.join(self._output_dir, filename + '.wav')
            manifest_list = []
            prev_speaker = None

            session_length_sr = int((self._session_length*self._sr))
            array = np.zeros(session_length_sr)

            while (running_length < session_length_sr):
                #select speaker
                speaker_turn = self.get_speaker(prev_speaker)

                #select speaker length
                sl = np.random.negative_binomial(self._sentence_length_params[0], self._sentence_length_params[1])
                sl_sr = int(sl*self._sr)

                #ensure session length is as desired
                if running_length+sl_sr > session_length_sr:
                    sl_sr = session_length_sr - running_length

                # only add if remaining length > 1 second
                if sl < 1:
                    break

                #load audio file
                file = self.load_speaker_sample(speaker_lists, speaker_ids, speaker_turn)
                audio_file, sr = librosa.load(file['audio_filepath'], sr=self._sr)
                sentence_duration = int(file['duration']*self._sr)
                sentence = np.zeros(sl_sr)

                #text, words, alignments
                text = ""
                words = []
                alignments = []

                while (sentence_duration < sl):
                    #copy sentence
                    begin = sentence_duration - int(file['duration']*self._sr)
                    end = sentence_duration
                    sentence[begin:end] = audio_file

                    #combine text, words, alignments here
                    if text != "":
                        text += " "
                    text += file['text'] #deal with space here
                    i = 0
                    for i in range(0, len(file['words'])):
                        words.append(file['words'][i])
                        alignments.append(sentence_duration+file['alignments'][i])

                    #load next audio file
                    file = self.load_speaker_sample(speaker_lists, speaker_ids, speaker_turn)
                    audio_file, sr = librosa.load(file['audio_filepath'], sr=self._sr)
                    sentence_duration += int(file['duration']*self._sr)

                sentence_duration = sentence_duration - int(file['duration']*self._sr)
                remaining_duration = sl_sr - sentence_duration

                # only add more words if remaining_duration > 1 second
                if remaining_duration > self._sr:
                    #use alignments to pad sentence
                    words = file['words']
                    alignments = file['alignments']
                    dur = 0
                    i = 0
                    dur = int(alignments[i]*self._sr)
                    prev_dur = 0
                    while (dur < remaining_duration):
                        word = words[i]
                        prev_dur = dur
                        #TODO append word and alignment here (and to text)
                        text += " " + word
                        words.append(file['words'][i])
                        alignments.append(sentence_duration+file['alignments'][i])
                        i += 1
                        dur = int(alignments[i]*self._sr)
                        print(i)
                        print(dur)
                    sentence[sentence_duration:sentence_duration+prev_dur] = audio_file[:prev_dur]

                start = running_length
                length = sl_sr

                # add overlap (currently overlapping with some frequency and with a maximum percentage of overlap)
                # also don't overlap same speaker

                #TODO also add silence

                end = start+length
                array[start:end] = audio_file[:length]

                new_entry = self.create_new_rttm_entry(file, running_length, speaker_ids[speaker_turn])
                manifest_list.append(new_entry)

                running_length += duration
                prev_speaker = speaker_turn

            sf.write(wavpath, array, self._sr)
            labels_to_rttmfile(manifest_list, filename, self._output_dir)
