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
import warnings

import librosa
import numpy as np
from scipy.stats import halfnorm
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

def write_manifest(output_path, target_manifest):
    with open(output_path, "w") as outfile:
        for tgt in target_manifest:
            json.dump(tgt, outfile)
            outfile.write('\n')

def write_ctm(output_path, target_ctm):
    target_ctm.sort(key=lambda y: y[0])
    with open(output_path, "w") as outfile:
        for pair in target_ctm:
            tgt = pair[1]
            outfile.write(tgt)

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
        alignment_type (str): input alignment format
                              end - end alignments passed
                              start - start alignments passed
                              tuple - alignments expected in (start,end) pairs
        dominance_var (float): variance in speaker dominance
        min_dominance (float): minimum percentage of speaking time per speaker
        turn_prob (float): probability of switching speakers
        mean_overlap (float): mean proportion of overlap to speaking time
        mean_silence (float): mean proportion of silence to speaking time
        outputs (str): which files to output (r - rttm, j - json, c - ctm)
        enforce_num_speakers (bool): enforce that all requested speakers are present in the output wav file
    """

    def __init__(self, cfg):
        self._params = cfg

        # internal params
        self._manifest = read_manifest(self._params.data_simulator.manifest_path)
        self._sentence = None
        self._text = ""
        self._words = []
        self._alignments = []

        #keep track of furthest sample per speaker to avoid overlapping same speaker
        self._furthest_sample = [0 for n in range(0,self._params.data_simulator.num_speakers)]

    # randomly select speaker ids from loaded dict
    def _get_speaker_ids(self):
        speaker_ids = []
        s = 0
        while s < self._params.data_simulator.num_speakers:
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
        for i in range(0, self._params.data_simulator.num_speakers):
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

    # get dominance for each speaker
    def _get_speaker_dominance(self):
        dominance_mean = 1.0/self._params.data_simulator.num_speakers
        dominance = np.random.normal(loc=dominance_mean, scale=self._params.data_simulator.dominance_var, size=self._params.data_simulator.num_speakers)
        for i in range(0,len(dominance)):
          if dominance[i] < 0:
            dominance[i] = 0
        #normalize while maintaining minimum dominance
        total = np.sum(dominance)
        if total == 0:
          for i in range(0,len(dominance)):
            dominance[i]+=min_dominance
        dominance = (dominance / total)*(1-self._params.data_simulator.min_dominance*self._params.data_simulator.num_speakers)
        for i in range(0,len(dominance)):
          dominance[i]+=self._params.data_simulator.min_dominance
          if i > 0:
            dominance[i] = dominance[i] + dominance[i-1]
        return dominance

    def _increase_speaker_dominance(self, increase_percent, base_speaker_dominance, factor):
        dominance = np.copy(base_speaker_dominance)
        for i in range(len(dominance)-1,0,-1):
            dominance[i] = dominance[i] - dominance[i-1]
        for i in increase_percent:
            dominance[i] = dominance[i] * factor
        dominance = dominance / np.sum(dominance)
        for i in range(1,len(dominance)):
            dominance[i] = dominance[i] + dominance[i-1]
        return dominance

    # get next speaker (accounting for turn probability, dominance distribution)
    def _get_next_speaker(self, prev_speaker, dominance):
        if random.uniform(0, 1) > self._params.data_simulator.turn_prob and prev_speaker != None:
            return prev_speaker
        else:
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
    def _add_file(self, file, audio_file, sentence_duration, max_sentence_duration, max_sentence_duration_sr):
        sentence_duration_sr = len(self._sentence)
        remaining_duration_sr = max_sentence_duration_sr - sentence_duration_sr
        remaining_duration = max_sentence_duration - sentence_duration
        prev_dur_sr = dur_sr = 0
        nw = i = 0

        #ensure the desired number of words are added and the length of the output session isn't exceeded
        while (nw < remaining_duration and dur_sr < remaining_duration_sr and i < len(file['words'])):
            dur_sr = int(file['alignments'][i] * self._params.data_simulator.sr)
            if dur_sr > remaining_duration_sr:
                break

            word = file['words'][i]
            self._words.append(word)

            if self._params.data_simulator.alignment_type == 'start':
                self._alignments.append(int(sentence_duration_sr / self._params.data_simulator.sr) + file['alignments'][i])
            elif self._params.data_simulator.alignment_type == 'end':
                self._alignments.append(int(sentence_duration_sr / self._params.data_simulator.sr) + file['alignments'][i])
            elif self._params.data_simulator.alignment_type == 'tuple':
                start = int(sentence_duration_sr / self._params.data_simulator.sr) + file['alignments'][i][0]
                end = int(sentence_duration_sr / self._params.data_simulator.sr) + file['alignments'][i][1]
                self._alignments.append((start,end))

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
    def _add_silence_or_overlap(self, speaker_turn, prev_speaker, start, length, session_length_sr, prev_length_sr, enforce):
        overlap_prob = self._params.data_simulator.overlap_prob / (self._params.data_simulator.turn_prob)  # accounting for not overlapping the same speaker
        mean_overlap_percent = self._params.data_simulator.mean_overlap / self._params.data_simulator.overlap_prob
        mean_silence_percent = self._params.data_simulator.mean_silence / (1 - self._params.data_simulator.overlap_prob)

        # overlap
        if prev_speaker != speaker_turn and prev_speaker != None and np.random.uniform(0, 1) < overlap_prob:
            overlap_percent = halfnorm(loc=0, scale=mean_overlap_percent*np.sqrt(np.pi)/np.sqrt(2)).rvs()
            if (overlap_percent > 1):
                overlap_percent = 1
            new_start = start - int(prev_length_sr * overlap_percent)
            #if same speaker ends up overlapping, pad with silence instead
            if (new_start < self._furthest_sample[speaker_turn]):
                new_start = self._furthest_sample[speaker_turn]
                #TODO should silence be added here
                silence_percent = mean_silence_percent + np.random.uniform(-mean_silence_percent, mean_silence_percent)
                silence_amount = int(length * silence_percent)
                if new_start + length + silence_amount > session_length_sr:
                    return session_length_sr - length
                else:
                    return new_start + silence_amount
            else:
                return new_start
        else:
            # add silence
            silence_percent = halfnorm(loc=0, scale=mean_silence_percent*np.sqrt(np.pi)/np.sqrt(2)).rvs()
            if (silence_percent > 1):
                silence_percent = 1
            silence_amount = int(length * silence_percent)

            if start + length + silence_amount > session_length_sr and not enforce:
                return session_length_sr - length
            else:
                return start + silence_amount

    # add new entry to dict (to write to output rttm file)
    def _create_new_rttm_entry(self, start, dur, speaker_id):
        start = float(round(start,3))
        dur = float(round(dur,3))
        return str(start) + ' ' + str(dur) + ' ' + str(speaker_id)

    # add new entry to dict (to write to output json file)
    def _create_new_json_entry(self, wav_filename, start, dur, speaker_id, text, rttm_filepath, ctm_filepath):
        start = float(round(start,3))
        dur = float(round(dur,3))
        dict = {"audio_filepath": wav_filename,
                "offset": start,
                "duration": dur,
                "label": speaker_id,
                "text": text,
                "num_speakers": self._params.data_simulator.num_speakers,
                "rttm_filepath": rttm_filepath,
                "ctm_filepath": ctm_filepath,
                "uem_filepath": None}
        return dict

    # add new entry to dict (to write to output ctm file)
    def _create_new_ctm_entry(self, session_name, speaker_id, start):
        arr = []
        start = float(round(start,3))
        for i in range(0, len(self._words)):
            word = self._words[i]
            if self._params.data_simulator.alignment_type == 'start':
                align1 = float(round(self._alignments[i] + start, 3))
                align2 = float(round(self._alignments[i+1] - self._alignments[i], 3))
            elif self._params.data_simulator.alignment_type == 'end':
                align1 = float(round(self._alignments[i-1] + start, 3))
                align2 = float(round(self._alignments[i] - self._alignments[i-1], 3))
            elif self._params.data_simulator.alignment_type == 'tuple':
                align1 = float(round(self._alignments[i][0] + start, 3))
                align2 = float(round(self._alignments[i][1] - self._alignments[i][0], 3))
            if word != "": #note that using the current alignments the first word is always empty, so there is no error from indexing the array with i-1
                text = str(session_name) + ' ' + str(speaker_id) + ' ' + str(align1) + ' ' + str(align2) + ' ' + str(word) + ' ' + '0' + '\n'
                arr.append((align1, text))
        return arr

    """
    Generate diarization session

    Args:
        num_sessions (int): number of diarization sessions to generate with the current configuration
    """

    def generate_session(self, num_sessions=1, random_seed=42):
        random.seed(random_seed)
        for i in range(0, num_sessions):
            speaker_ids = self._get_speaker_ids()  # randomly select speaker ids
            speaker_dominance = self._get_speaker_dominance()  # randomly determine speaker dominance
            base_speaker_dominance = np.copy(speaker_dominance)
            speaker_lists = self._get_speaker_samples(speaker_ids)  # get list of samples per speaker

            filename = self._params.data_simulator.output_filename + f"_{i}"
            wavpath = os.path.join(self._params.data_simulator.output_dir, filename + '.wav')
            speaker_turn = 0  # assume alternating between speakers 1 & 2
            running_length_sr = 0  # starting point for each sentence
            prev_length_sr = 0  # for overlap
            start = end = 0
            prev_speaker = None
            rttm_list = []
            json_list = []
            ctm_list = []
            self._furthest_sample = [0 for n in range(0,self._params.data_simulator.num_speakers)]

            #hold enforce until all speakers have spoken
            enforce_counter = 2
            enforce_time = random.uniform(0.25, 0.75)
            if self._params.data_simulator.enforce_num_speakers:
                enforce = True
            else:
                enforce = False

            ROOT = os.getcwd()
            rttm_filepath = os.path.join(ROOT, self._params.data_simulator.output_dir, filename + '.rttm')
            json_filepath = os.path.join(ROOT, self._params.data_simulator.output_dir, filename + '.json')
            ctm_filepath = os.path.join(ROOT, self._params.data_simulator.output_dir, filename + '.ctm')

            session_length_sr = int((self._params.data_simulator.session_length * self._params.data_simulator.sr))
            array = np.zeros(session_length_sr)

            while running_length_sr < session_length_sr or enforce:
                #enforce num_speakers
                if running_length_sr > enforce_time*session_length_sr and enforce:
                    increase_percent = []
                    for i in range(0,self._params.data_simulator.num_speakers):
                        if self._furthest_sample[i] == 0:
                            increase_percent.append(i)
                    #ramp up enforce counter until speaker is sampled, then reset once all speakers have spoken
                    if len(increase_percent) > 0:
                        speaker_dominance = self._increase_speaker_dominance(increase_percent, base_speaker_dominance, enforce_counter)
                        enforce_counter += 1
                    else:
                        enforce = False
                        speaker_dominance = base_speaker_dominance


                # select speaker
                speaker_turn = self._get_next_speaker(prev_speaker, speaker_dominance)

                # select speaker length
                sl = np.random.negative_binomial(
                    self._params.data_simulator.sentence_length_params[0], self._params.data_simulator.sentence_length_params[1]
                )
                max_sentence_duration_sr = session_length_sr - running_length_sr

                # only add if remaining length > 0.5 second
                if max_sentence_duration_sr < 0.5 * self._params.data_simulator.sr and not enforce:
                    break
                if enforce:
                    max_sentence_duration_sr = float('inf')

                # initialize sentence, text, words, alignments
                self._sentence = np.zeros(0)
                self._text = ""
                self._words = []
                self._alignments = []
                sentence_duration = sentence_duration_sr = 0

                # build sentence
                while sentence_duration < sl and sentence_duration_sr < max_sentence_duration_sr:
                    file = self._load_speaker_sample(speaker_lists, speaker_ids, speaker_turn)
                    audio_file, sr = librosa.load(file['audio_filepath'], sr=self._params.data_simulator.sr)
                    sentence_duration,sentence_duration_sr = self._add_file(file, audio_file, sentence_duration, sl, max_sentence_duration_sr)

                length = len(self._sentence)
                start = self._add_silence_or_overlap(
                    speaker_turn, prev_speaker, running_length_sr, length, session_length_sr, prev_length_sr, enforce
                )
                end = start + length
                if end > len(array):
                    array = np.pad(array, (0, end - len(array)))
                array[start:end] += self._sentence

                if 'r' in self._params.data_simulator.outputs:
                    new_rttm_entry = self._create_new_rttm_entry(start / self._params.data_simulator.sr, end / self._params.data_simulator.sr, speaker_ids[speaker_turn])
                    rttm_list.append(new_rttm_entry)
                if 'j' in self._params.data_simulator.outputs:
                    new_json_entry = self._create_new_json_entry(wavpath, start / self._params.data_simulator.sr, length / self._params.data_simulator.sr, speaker_ids[speaker_turn], self._text, rttm_filepath, ctm_filepath)
                    json_list.append(new_json_entry)
                if 'c' in self._params.data_simulator.outputs:
                    new_ctm_entries = self._create_new_ctm_entry(filename, speaker_ids[speaker_turn], start / self._params.data_simulator.sr)
                    for entry in new_ctm_entries:
                        ctm_list.append(entry)

                running_length_sr = np.maximum(running_length_sr, end)
                self._furthest_sample[speaker_turn] = running_length_sr
                prev_speaker = speaker_turn
                prev_length_sr = length

            k = np.sum(self._furthest_sample == 0)
            if k != 0:
                warnings.warn(f"{self._params.data_simulator.num_speakers-k} speakers were included in the clip instead of the requested amount of {self._params.data_simulator.num_speakers}")

            array = array / (1.0 * np.max(np.abs(array)))  # normalize wav file
            sf.write(wavpath, array, self._params.data_simulator.sr)
            if 'r' in self._params.data_simulator.outputs:
                labels_to_rttmfile(rttm_list, filename, self._params.data_simulator.output_dir)
            if 'j' in self._params.data_simulator.outputs:
                write_manifest(json_filepath, json_list)
            if 'c' in self._params.data_simulator.outputs:
                write_ctm(ctm_filepath, ctm_list)
