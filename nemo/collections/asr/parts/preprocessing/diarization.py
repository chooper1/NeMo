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

import json
import os
import random

import librosa
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

from nemo.collections.asr.parts.utils.speaker_utils import labels_to_rttmfile


# from scripts/speaker_tasks/filelist_to_manifest.py - move function?
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
                              initial values are from page 209 of
                              https://www.researchgate.net/publication/318396023_How_will_text_size_influence_the_length_of_its_linguistic_constituents
        dominance_dist (str): same - same probability for each speakers
                              random - random probabilities for each speaker
        turn_prob (float): probability of switching speakers
        mean_overlap (float): mean proportion of overlap to speaking time
        mean_silence (float): mean proportion of silence to speaking time
    """

    def __init__(
        self,
        manifest_path=None,
        sr=16000,
        num_speakers=2,
        session_length=60,
        output_dir='output',
        output_filename='librispeech_diarization',
        sentence_length_params=[2.81, 0.1],
        dominance_dist="random",
        turn_prob=0.9,
        mean_overlap=0.08,
        mean_silence=0.08,
        overlap_prob=0.3,
    ):
        self._manifest_path = manifest_path
        self._sr = sr
        self._num_speakers = num_speakers
        self._session_length = session_length
        self._output_dir = output_dir
        self._output_filename = output_filename
        self._sentence_length_params = sentence_length_params
        self._dominance_dist = dominance_dist
        self._turn_prob = turn_prob
        self._mean_overlap = mean_overlap
        self._mean_silence = mean_silence
        self._overlap_prob = overlap_prob

        # internal params
        self._manifest = read_manifest(manifest_path)
        self._sentence = None
        self._text = ""
        self._words = []
        self._alignments = []

    """
    Load all parameters from a config file (yaml)

    Args:
        config_path (str): path to a valid config file
    """

    def load_config(self, config_path):
        config = OmegaConf.load(config_path)
        self._manifest_path = config["manifest_path"]
        self._sr = config["sr"]
        self._num_speakers = config["num_speakers"]
        self._session_length = config["session_length"]
        self._output_dir = config["output_dir"]
        self._output_filename = config["output_filename"]
        self._sentence_length_params = config["sentence_length_params"]
        self._dominance_dist = config["dominance_dist"]
        self._turn_prob = config["turn_prob"]
        self._mean_overlap = config["mean_overlap"]
        self._mean_silence = config["mean_silence"]
        self._overlap_prob = config["overlap_prob"]

    """
    Write all parameters to a config file (yaml)

    Args:
        config_path (str): target path for the config file
    """

    def write_config(self, config_path):
        conf = OmegaConf.create(
            {
                "manifest_path": self._manifest_path,
                "sr": self._sr,
                "num_speakers": self._num_speakers,
                "session_length": self._session_length,
                "output_dir": self._output_dir,
                "output_filename": self._output_filename,
                "sentence_length_params": self._sentence_length_params,
                "dominance_dist": self._dominance_dist,
                "turn_prob": self._turn_prob,
                "mean_overlap": self._mean_overlap,
                "mean_silence": self._mean_silence,
                "overlap_prob": self._overlap_prob,
            }
        )
        OmegaConf.save(config=conf, f=config_path)

    # randomly select speaker ids from loaded dict
    def _get_speaker_ids(self):
        speaker_ids = []
        s = 0
        while s < self._num_speakers:
            file = self._manifest[random.randint(0, len(self._manifest) - 1)]
            fn = file['audio_filepath'].split('/')[-1]
            speaker_id = fn.split('-')[0]
            # ensure speaker ids are not duplicated
            if speaker_id not in speaker_ids:
                speaker_ids.append(speaker_id)
                s += 1
        return speaker_ids

    # get a list of the samples for the specified speakers
    def _get_speaker_samples(self, speaker_ids):
        speaker_lists = {}
        for i in range(0, self._num_speakers):
            spid = speaker_ids[i]
            speaker_lists[str(spid)] = []

        for file in self._manifest:
            fn = file['audio_filepath'].split('/')[-1]
            new_speaker_id = fn.split('-')[0]
            for spid in speaker_ids:
                if spid == new_speaker_id:
                    speaker_lists[str(spid)].append(file)

        return speaker_lists

    # load a sample for the selected speaker id
    def _load_speaker_sample(self, speaker_lists, speaker_ids, speaker_turn):
        speaker_id = speaker_ids[speaker_turn]
        file_id = random.randint(0, len(speaker_lists[str(speaker_id)]) - 1)
        file = speaker_lists[str(speaker_id)][file_id]
        return file

    # add new entry to dict (to write to output manifest file)
    def _create_new_rttm_entry(self, start, dur, speaker_id):
        return str(start) + ' ' + str(dur) + ' ' + str(speaker_id)

    # get dominance for each speaker
    def _get_speaker_dominance(self):
        dominance = None
        # set n-1 random thresholds to get a variable speaker distribution
        if self._dominance_dist == "random":
            dominance = [random.uniform(0, 1) for s in range(0, self._num_speakers - 1)]
            dominance.sort()
            dominance.append(1)
        return dominance

    # get next speaker (accounting for turn probability, dominance distribution)
    def _get_next_speaker(self, prev_speaker, dominance):
        if random.uniform(0, 1) > self._turn_prob and prev_speaker != None:
            return prev_speaker

        if self._dominance_dist == "same":
            speaker_turn = random.randint(0, self._num_speakers - 1)
            while speaker_turn == prev_speaker:
                speaker_turn = random.randint(0, self._num_speakers - 1)

        elif self._dominance_dist == "random":
            rand = random.uniform(0, 1)
            speaker_turn = 0
            while rand > dominance[speaker_turn]:
                speaker_turn += 1
            while speaker_turn == prev_speaker:
                rand = random.uniform(0, 1)
                speaker_turn = 0
                while rand > dominance[speaker_turn]:
                    speaker_turn += 1

        return speaker_turn

    # add audio file to current sentence
    # TODO ensure session length is as desired (clip sentence length at end)
    def _add_file(self, file, audio_file, sentence_duration, max_sentence_duration, max_sentence_duration_sr):
        #get number of words
        num_words = len([word for word in file['words'] if word != ""])
        sentence_duration_sr = len(self._sentence)
        # enough room to add the entire audio file
        if num_words < max_sentence_duration - sentence_duration and sentence_duration_sr + len(audio_file) <= max_sentence_duration_sr:
            self._sentence = np.append(self._sentence, audio_file)
            # combine text, words, alignments here
            if self._text != "":
                self._text += " "
            self._text += file['text']
            self._words += file['words']
            for i in range(0, len(file['words'])):
                self._alignments.append(int(sentence_duration_sr / self._sr) + file['alignments'][i])
            return sentence_duration+num_words, sentence_duration_sr+len(audio_file)
        #not enough room to add the entire audio file or not all words are needed
        else:
            remaining_duration_sr = max_sentence_duration_sr - sentence_duration_sr
            remaining_duration = max_sentence_duration - sentence_duration
            prev_dur_sr = dur_sr = 0
            nw = i = 0
            while nw < remaining_duration and dur_sr < remaining_duration_sr and i < len(file['words']):
                dur_sr = int(file['alignments'][i] * self._sr)
                if dur_sr > remaining_duration_sr:
                    break

                word = file['words'][i]
                self._words.append(word)
                self._alignments.append(int(sentence_duration_sr / self._sr) + file['alignments'][i])
                if word == "":
                    i+=1
                    continue
                elif self._text == "":
                    self._text += word
                else:
                    self._text += " " + word
                i+=1
                nw+=1
                prev_dur_sr = dur_sr
            # add audio clip up to the final alignment
            self._sentence = np.append(self._sentence, audio_file[:prev_dur_sr])
            if dur_sr > remaining_duration_sr:
                self._sentence = np.pad(self._sentence, (0, max_sentence_duration_sr - len(self._sentence)))
            return sentence_duration+nw, len(self._sentence)

    # returns new overlapped (or shifted) start position
    def _add_silence_or_overlap(self, speaker_turn, prev_speaker, start, length, session_length_sr, prev_length_sr):
        overlap_prob = self._overlap_prob / (self._turn_prob)  # accounting for not overlapping the same speaker
        mean_overlap_percent = self._mean_overlap / self._overlap_prob
        mean_silence_percent = self._mean_silence / (1 - self._overlap_prob)

        # overlap
        if prev_speaker != speaker_turn and prev_speaker != None:
            if np.random.uniform(0, 1) < overlap_prob:
                overlap_percent = mean_overlap_percent + np.random.uniform(-mean_overlap_percent, mean_overlap_percent)
                return start - int(prev_length_sr * overlap_percent)

        # add silence
        silence_percent = mean_silence_percent + np.random.uniform(-mean_silence_percent, mean_silence_percent)
        silence_amount = int(length * silence_percent)
        if start + length + silence_amount > session_length_sr:
            return session_length_sr - length
        else:
            return start + silence_amount

    """
    Generate diarization session

    Args:
        num_sessions (int): number of diarization sessions to generate with the current configuration
    """

    def generate_session(self, num_sessions=1):
        for i in range(0, num_sessions):
            speaker_ids = self._get_speaker_ids()  # randomly select speaker ids
            speaker_dominance = self._get_speaker_dominance()  # randomly determine speaker dominance
            speaker_lists = self._get_speaker_samples(speaker_ids)  # get list of samples per speaker

            filename = self._output_filename + f"_{i}"
            wavpath = os.path.join(self._output_dir, filename + '.wav')
            speaker_turn = 0  # assume alternating between speakers 1 & 2
            running_length_sr = 0  # starting point for each sentence
            prev_length_sr = 0  # for overlap
            start = end = 0
            prev_speaker = None
            manifest_list = []

            session_length_sr = int((self._session_length * self._sr))
            array = np.zeros(session_length_sr)

            while running_length_sr < session_length_sr:
                # select speaker
                speaker_turn = self._get_next_speaker(prev_speaker, speaker_dominance)

                # select speaker length
                sl = np.random.negative_binomial(
                    self._sentence_length_params[0], self._sentence_length_params[1]
                )
                max_sentence_duration_sr = session_length_sr - running_length_sr

                # only add if remaining length > 0.5 second
                if session_length_sr - running_length_sr < 0.5 * self._sr:
                    break

                # initialize sentence, text, words, alignments
                self._sentence = np.zeros(0)
                self._text = ""
                self._words = []
                self._alignments = []
                sentence_duration = sentence_duration_sr = 0

                # build sentence
                while sentence_duration < sl and sentence_duration_sr < max_sentence_duration_sr:
                    file = self._load_speaker_sample(speaker_lists, speaker_ids, speaker_turn)
                    audio_file, sr = librosa.load(file['audio_filepath'], sr=self._sr)
                    sentence_duration,sentence_duration_sr = self._add_file(file, audio_file, sentence_duration, sl, max_sentence_duration_sr)

                length = len(self._sentence)
                # add overlap or silence
                start = self._add_silence_or_overlap(
                    speaker_turn, prev_speaker, running_length_sr, length, session_length_sr, prev_length_sr
                )
                end = start + length
                array[start:end] = self._sentence

                new_entry = self._create_new_rttm_entry(start / self._sr, end / self._sr, speaker_ids[speaker_turn])
                manifest_list.append(new_entry)

                running_length_sr = np.maximum(running_length_sr, end)
                prev_speaker = speaker_turn
                prev_length_sr = length

            array = array / (1.0 * np.max(np.abs(array)))  # normalize wav file
            sf.write(wavpath, array, self._sr)
            labels_to_rttmfile(manifest_list, filename, self._output_dir)
