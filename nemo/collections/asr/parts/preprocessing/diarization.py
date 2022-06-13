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

from omegaconf import OmegaConf

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
        sentence_length_params (list): k,p values for negative_binomial distribution
        dominance_dist (str): same - same probability for each speakers
                              random - random probabilities for each speaker
        turn_prob (float): probability of switching speakers
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
        dominance_dist = "random",
        turn_prob = 0.9,
    ):
        self._manifest_path = manifest_path
        self._sr = sr
        self._num_speakers = num_speakers
        self._session_length = session_length
        self._output_dir = output_dir
        self._output_filename = output_filename

        self._sentence_length_params = sentence_length_params

        #dominance distribution:
        #same - same for each speaker
        #rand - randomly distributed (pick num_speaker random uniform values)
        self._dominance_dist = dominance_dist
        self._turn_prob = turn_prob

        #overlap/silence


        #internal params
        self._manifest = read_manifest(manifest_path)
        self._sentence = None
        self._text = ""
        self._words = []
        self._alignments = []

        self._config_path = None

    #Get/Set Methods
    def set_session_length(self, new_sl):
        self._session_length = new_sl

    def set_output_filename(self, new_fn):
        self._output_filename = new_fn

    def set_num_speakers(self, new_ns):
        self._num_speakers = new_ns

    # load all parameters from a config file (yaml)
    def load_config(self, config_path):
        self._config_path = config_path
        config = OmegaConf.load(config_path)
        print(OmegaConf.to_yaml(config))

    def write_config(self, config_path):
        self._config_path = config_path

        file = OmegaConf.create({"manifest_path": self._manifest_path,
                                "sr": self._sr,
                                "num_speakers": self._num_speakers,
                                "session_length": self._session_length,
                                "output_dir": self._output_dir,
                                "output_filename": self._output_filename,
                                "sentence_length_params": self._sentence_length_params,
                                "dominance_dist": self._dominance_dist,
                                "turn_prob": self._turn_prob,
                                "sr": self._sr})
        OmegaConf.save(config=conf, f=config_path)

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
    def create_new_rttm_entry(self, start, dur, speaker_id):
        return str(start) + ' ' + str(dur) + ' ' + str(speaker_id)

    #get dominance for each speaker
    def get_speaker_dominance(self):
        dominance = None
        # if self._dominance_dist == "same":
        #     dominance_factor = 1.0/self._num_speakers
        #     dominance = [s*dominance_factor for s in range(1, self._num_speakers + 1)]
        if self._dominance_dist == "random":
            dominance = [random.uniform(0, 1) for s in range(0, self._num_speakers - 1)]
            dominance.sort()
            dominance.append(1)
        print(dominance)
        return dominance

    #sample from speakers
    #TODO account for speaker dominance
    def get_speaker(self, prev_speaker, dominance):
        if self._dominance_dist == "same":
            speaker_turn = random.randint(0,self._num_speakers-1)
            if (prev_speaker != None):
                if (random.uniform(0, 1) < self._turn_prob):
                    while (speaker_turn == prev_speaker):
                        speaker_turn = random.randint(0, self._num_speakers-1)
                else:
                    speaker_turn = prev_speaker

        elif self._dominance_dist == "random":
            rand = random.uniform(0, 1)
            speaker_turn = 0
            while rand > dominance[speaker_turn]:
                speaker_turn += 1

            if (prev_speaker != None):
                if (random.uniform(0, 1) < self._turn_prob):
                    while (speaker_turn == prev_speaker):
                        rand = random.uniform(0, 1)
                        speaker_turn = 0
                        while rand > dominance[speaker_turn]:
                            speaker_turn += 1
                else:
                    speaker_turn = prev_speaker

        return speaker_turn

    #add audio file to current sentence
    def add_file(self, file, audio_file, sentence_duration_sr, max_sentence_duration_sr):
        #add to self._sentence
        if (sentence_duration_sr + len(audio_file) < max_sentence_duration_sr):
            begin = sentence_duration_sr
            end = sentence_duration_sr + len(audio_file)
            self._sentence[begin:end] = audio_file

            #combine text, words, alignments here
            if self._text != "":
                self._text += " "
            self._text += file['text']
            i = 0
            for i in range(0, len(file['words'])):
                self._words.append(file['words'][i])
                self._alignments.append(int(sentence_duration_sr/self._sr)+file['alignments'][i])

            sentence_duration_sr += len(audio_file)
            return sentence_duration_sr

        elif max_sentence_duration_sr - sentence_duration_sr > 0.5*self._sr:
            #atleast 0.5 second remaining in sentence - use alignments to pad sentence
            remaining_duration = max_sentence_duration_sr - sentence_duration_sr
            dur = prev_dur = 0
            for i in range(0,len(file['words'])):
                dur = int(file['alignments'][i]*self._sr)
                if dur > remaining_duration:
                    break
                else:
                    word = file['words'][i]
                    if self._text == "":
                        self._text += word
                    elif word != "":
                        self._text += " " + word
                    self._words.append(word)
                    self._alignments.append(int(sentence_duration_sr/self._sr)+file['alignments'][i])
                    prev_dur = dur
            if prev_dur > 0:
                self._sentence[sentence_duration_sr:sentence_duration_sr+prev_dur] = audio_file[:prev_dur]

            return max_sentence_duration_sr

        else:
            return max_sentence_duration_sr


    #generate diarization session
    def generate_session(self, num_sessions=1):
        for i in range(0,num_sessions):
            filename = self._output_filename + f"_{i}"

            speaker_ids = self.get_speaker_ids() #randomly select speaker ids
            speaker_dominance = self.get_speaker_dominance() #randomly determine speaker dominance
            speaker_lists = self.get_speaker_samples(speaker_ids) #get list of samples per speaker

            speaker_turn = 0 #assume alternating between speakers 1 & 2
            running_length_sr = 0 #starting point for each sentence
            previous_duration = 0 #for overlap

            wavpath = os.path.join(self._output_dir, filename + '.wav')
            manifest_list = []
            prev_speaker = None

            session_length_sr = int((self._session_length*self._sr))
            array = np.zeros(session_length_sr)

            while (running_length_sr < session_length_sr):
                #select speaker
                speaker_turn = self.get_speaker(prev_speaker, speaker_dominance)

                #select speaker length
                #TODO ensure length is atleast one word
                sl = np.random.negative_binomial(self._sentence_length_params[0], self._sentence_length_params[1])
                sl += random.uniform(-0.5, 0.5)
                if sl < 0:
                    sl = 0
                #inserting randomness into sentence length
                sl_sr = int(sl*self._sr)

                #ensure session length is as desired (clip sentence length at end)
                if running_length_sr+sl_sr > session_length_sr:
                    sl_sr = session_length_sr - running_length_sr

                # only add if remaining length > 0.5 second
                if session_length_sr-running_length_sr < 0.5*self._sr:
                    break

                #text, words, alignments
                self._text = ""
                self._words = []
                self._alignments = []

                self._sentence = np.zeros(sl_sr)
                sentence_duration = 0
                while (sentence_duration < sl_sr):
                    file = self.load_speaker_sample(speaker_lists, speaker_ids, speaker_turn)
                    audio_file, sr = librosa.load(file['audio_filepath'], sr=self._sr)
                    sentence_duration = self.add_file(file, audio_file, sentence_duration, sl_sr)

                start = running_length_sr
                length = sl_sr

                # add overlap (currently overlapping with some frequency and with a maximum percentage of overlap)
                # also don't overlap same speaker

                #TODO also add silence

                end = start+length
                array[start:end] = self._sentence #audio_file[:length]

                new_entry = self.create_new_rttm_entry(start/self._sr, end/self._sr, speaker_ids[speaker_turn])
                manifest_list.append(new_entry)

                running_length_sr += sl_sr
                prev_speaker = speaker_turn

            sf.write(wavpath, array, self._sr)
            labels_to_rttmfile(manifest_list, filename, self._output_dir)
