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

import collections
import json
import os
from itertools import combinations
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml

from nemo.collections.common.parts.preprocessing import manifest, parsers
# from nemo.collections.asr.parts.utils.speaker_utils import get_rttm_speaker_index, rttm_to_labels
from nemo.utils import logging
from nemo.collections.asr.parts.preprocessing import LibriSpeechGenerator

from collections import Counter
from collections import OrderedDict as od
from nemo.collections.asr.parts.utils.speaker_utils import (
    get_rttm_speaker_index,
    audio_rttm_map,
    get_subsegments,
    get_embs_and_timestamps,
    get_uniqname_from_filepath,
    parse_scale_configs,
    perform_clustering,
    score_labels,
    segments_manifest_to_subsegments_manifest,
    write_rttm2manifest,
    rttm_to_labels,
    labels_to_pyannote_object
)

def write_file(name, lines, idx):
    with open(name, 'w') as fout:
        for i in idx:
            dic = lines[i]
            json.dump(dic, fout)
            fout.write('\n')
    logging.info("wrote", name)


def read_file(pathlist):
    pathlist = open(pathlist, 'r').readlines()
    return sorted(pathlist)


def get_dict_from_wavlist(pathlist):
    path_dict = od()
    pathlist = sorted(pathlist)
    for line_path in pathlist:
        uniq_id = os.path.basename(line_path).split('.')[0]
        path_dict[uniq_id] = line_path
    return path_dict


def get_dict_from_list(data_pathlist, uniqids):
    path_dict = {}
    for line_path in data_pathlist:
        uniq_id = os.path.basename(line_path).split('.')[0]
        if uniq_id in uniqids:
            path_dict[uniq_id] = line_path
        else:
            raise ValueError(f'uniq id {uniq_id} is not in wav filelist')
    return path_dict


def get_path_dict(data_path, uniqids, len_wavs=None):
    if data_path is not None:
        data_pathlist = read_file(data_path)
        if len_wavs is not None:
            assert len(data_pathlist) == len_wavs
            data_pathdict = get_dict_from_list(data_pathlist, uniqids)
    elif len_wavs is not None:
        data_pathdict = {uniq_id: None for uniq_id in uniqids}
    return data_pathdict

def rreplace(s, old, new):
    li = s.rsplit(old, 1)
    return new.join(li)

def write_file(name, lines, idx):
    with open(name, 'w') as fout:
        for i in idx:
            dic = lines[i]
            json.dump(dic, fout)
            fout.write('\n')
    logging.info("wrote", name)

def get_uniq_id_with_period(path):
    split_path = os.path.basename(path).split('.')[:-1]
    uniq_id = '.'.join(split_path) if len(split_path) > 1 else split_path[0]
    return uniq_id

def get_subsegment_dict(subsegments_manifest_file, window, shift, deci):
    _subsegment_dict = {}
    with open(subsegments_manifest_file, 'r') as subsegments_manifest:
        print(f"Reading subsegments_manifest_file: {subsegments_manifest_file}")
        segments = subsegments_manifest.readlines()
        for segment in segments:
            segment = segment.strip()
            dic = json.loads(segment)
            audio, offset, duration, label = dic['audio_filepath'], dic['offset'], dic['duration'], dic['label']
            subsegments = get_subsegments(offset=offset, window=window, shift=shift, duration=duration)
            uniq_id = get_uniq_id_with_period(audio)
            if uniq_id not in _subsegment_dict:
                _subsegment_dict[uniq_id] = {'ts' : [], 'json_dic': []}
            for subsegment in subsegments:
                start, dur = subsegment
            _subsegment_dict[uniq_id]['ts'].append([round(start, deci), round(start+dur, deci)])
            _subsegment_dict[uniq_id]['json_dic'].append(dic)
    return _subsegment_dict

def get_input_manifest_dict(input_manifest_path):
    input_manifest_dict = {}
    with open(input_manifest_path, 'r') as input_manifest_fp:
        json_lines = input_manifest_fp.readlines()
        for json_line in json_lines:
            dic = json.loads(json_line)
            dic["text"] = "-"
            uniq_id = get_uniqname_from_filepath(dic["audio_filepath"])
            input_manifest_dict[uniq_id] = dic
    return input_manifest_dict

def write_truncated_subsegments(input_manifest_dict, _subsegment_dict, output_manifest_path, step_count, deci):
    with open(output_manifest_path, 'w') as output_manifest_fp:
        for uniq_id, subseg_dict in _subsegment_dict.items():
            print(f"Writing {uniq_id}")
            subseg_array = np.array(subseg_dict['ts'])
            subseg_array_idx = np.argsort(subseg_array, axis=0)
            chunked_set_count = subseg_array_idx.shape[0] // step_count

            for idx in range(chunked_set_count-1):
                chunk_index_stt = subseg_array_idx[:, 0][idx * step_count]
                chunk_index_end = subseg_array_idx[:, 1][(idx+1)* step_count]
                offset_sec = subseg_array[chunk_index_stt, 0]
                end_sec = subseg_array[chunk_index_end, 1]
                dur = round(end_sec - offset_sec, deci)
                meta = input_manifest_dict[uniq_id]
                meta['offset'] = offset_sec
                meta['duration'] = dur
                json.dump(meta, output_manifest_fp)
                output_manifest_fp.write("\n")

class _Collection(collections.UserList):
    """List of parsed and preprocessed data."""

    OUTPUT_TYPE = None  # Single element output type.


class Text(_Collection):
    """Simple list of preprocessed text entries, result in list of tokens."""

    OUTPUT_TYPE = collections.namedtuple('TextEntity', 'tokens')

    def __init__(self, texts: List[str], parser: parsers.CharParser):
        """Instantiates text manifest and do the preprocessing step.

        Args:
            texts: List of raw texts strings.
            parser: Instance of `CharParser` to convert string to tokens.
        """

        data, output_type = [], self.OUTPUT_TYPE
        for text in texts:
            tokens = parser(text)

            if tokens is None:
                logging.warning("Fail to parse '%s' text line.", text)
                continue

            data.append(output_type(tokens))

        super().__init__(data)


class FromFileText(Text):
    """Another form of texts manifest with reading from file."""

    def __init__(self, file: str, parser: parsers.CharParser):
        """Instantiates text manifest and do the preprocessing step.

        Args:
            file: File path to read from.
            parser: Instance of `CharParser` to convert string to tokens.
        """

        texts = self.__parse_texts(file)

        super().__init__(texts, parser)

    @staticmethod
    def __parse_texts(file: str) -> List[str]:
        if not os.path.exists(file):
            raise ValueError('Provided texts file does not exists!')

        _, ext = os.path.splitext(file)
        if ext == '.csv':
            texts = pd.read_csv(file)['transcript'].tolist()
        elif ext == '.json':  # Not really a correct json.
            texts = list(item['text'] for item in manifest.item_iter(file))
        else:
            with open(file, 'r') as f:
                texts = f.readlines()

        return texts


class AudioText(_Collection):
    """List of audio-transcript text correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(
        typename='AudioTextEntity',
        field_names='id audio_file duration text_tokens offset text_raw speaker orig_sr lang',
    )

    def __init__(
        self,
        ids: List[int],
        audio_files: List[str],
        durations: List[float],
        texts: List[str],
        offsets: List[str],
        speakers: List[Optional[int]],
        orig_sampling_rates: List[Optional[int]],
        token_labels: List[Optional[int]],
        langs: List[Optional[str]],
        parser: parsers.CharParser,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_number: Optional[int] = None,
        do_sort_by_duration: bool = False,
        index_by_file_id: bool = False,
    ):
        """Instantiates audio-text manifest with filters and preprocessing.

        Args:
            ids: List of examples positions.
            audio_files: List of audio files.
            durations: List of float durations.
            texts: List of raw text transcripts.
            offsets: List of duration offsets or None.
            speakers: List of optional speakers ids.
            orig_sampling_rates: List of original sampling rates of audio files.
            langs: List of language ids, one for eadh sample, or None.
            parser: Instance of `CharParser` to convert string to tokens.
            min_duration: Minimum duration to keep entry with (default: None).
            max_duration: Maximum duration to keep entry with (default: None).
            max_number: Maximum number of samples to collect.
            do_sort_by_duration: True if sort samples list by duration. Not compatible with index_by_file_id.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
        """

        output_type = self.OUTPUT_TYPE
        data, duration_filtered, num_filtered, total_duration = [], 0.0, 0, 0.0
        if index_by_file_id:
            self.mapping = {}

        for id_, audio_file, duration, offset, text, speaker, orig_sr, token_labels, lang in zip(
            ids, audio_files, durations, offsets, texts, speakers, orig_sampling_rates, token_labels, langs
        ):
            # Duration filters.
            if min_duration is not None and duration < min_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            if max_duration is not None and duration > max_duration:
                duration_filtered += duration
                num_filtered += 1
                continue

            if token_labels is not None:
                text_tokens = token_labels
            else:
                if text != '':
                    if hasattr(parser, "is_aggregate") and parser.is_aggregate:
                        if lang is not None:
                            text_tokens = parser(text, lang)
                        else:
                            raise ValueError("lang required in manifest when using aggregate tokenizers")
                    else:
                        text_tokens = parser(text)
                else:
                    text_tokens = []

                if text_tokens is None:
                    duration_filtered += duration
                    num_filtered += 1
                    continue

            total_duration += duration

            data.append(output_type(id_, audio_file, duration, text_tokens, offset, text, speaker, orig_sr, lang))
            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(audio_file))
                if file_id not in self.mapping:
                    self.mapping[file_id] = []
                self.mapping[file_id].append(len(data) - 1)

            # Max number of entities filter.
            if len(data) == max_number:
                break

        if do_sort_by_duration:
            if index_by_file_id:
                logging.warning("Tried to sort dataset by duration, but cannot since index_by_file_id is set.")
            else:
                data.sort(key=lambda entity: entity.duration)

        logging.info("Dataset loaded with %d files totalling %.2f hours", len(data), total_duration / 3600)
        logging.info("%d files were filtered totalling %.2f hours", num_filtered, duration_filtered / 3600)

        super().__init__(data)


class ASRAudioText(AudioText):
    """`AudioText` collector from asr structured json files."""

    def __init__(self, manifests_files: Union[str, List[str]], *args, **kwargs):
        """Parse lists of audio files, durations and transcripts texts.

        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            *args: Args to pass to `AudioText` constructor.
            **kwargs: Kwargs to pass to `AudioText` constructor.
        """

        ids, audio_files, durations, texts, offsets, = [], [], [], [], []
        speakers, orig_srs, token_labels, langs = [], [], [], []
        for item in manifest.item_iter(manifests_files):
            ids.append(item['id'])
            audio_files.append(item['audio_file'])
            durations.append(item['duration'])
            texts.append(item['text'])
            offsets.append(item['offset'])
            speakers.append(item['speaker'])
            orig_srs.append(item['orig_sr'])
            token_labels.append(item['token_labels'])
            langs.append(item['lang'])
        super().__init__(
            ids, audio_files, durations, texts, offsets, speakers, orig_srs, token_labels, langs, *args, **kwargs
        )


class SpeechLabel(_Collection):
    """List of audio-label correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(typename='SpeechLabelEntity', field_names='audio_file duration label offset',)

    def __init__(
        self,
        audio_files: List[str],
        durations: List[float],
        labels: List[Union[int, str]],
        offsets: List[Optional[float]],
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        max_number: Optional[int] = None,
        do_sort_by_duration: bool = False,
        index_by_file_id: bool = False,
    ):
        """Instantiates audio-label manifest with filters and preprocessing.

        Args:
            audio_files: List of audio files.
            durations: List of float durations.
            labels: List of labels.
            offsets: List of offsets or None.
            min_duration: Minimum duration to keep entry with (default: None).
            max_duration: Maximum duration to keep entry with (default: None).
            max_number: Maximum number of samples to collect.
            do_sort_by_duration: True if sort samples list by duration.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
        """

        if index_by_file_id:
            self.mapping = {}
        output_type = self.OUTPUT_TYPE
        data, duration_filtered = [], 0.0
        for audio_file, duration, command, offset in zip(audio_files, durations, labels, offsets):
            # Duration filters.
            if min_duration is not None and duration < min_duration:
                duration_filtered += duration
                continue

            if max_duration is not None and duration > max_duration:
                duration_filtered += duration
                continue

            data.append(output_type(audio_file, duration, command, offset))

            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(audio_file))
                self.mapping[file_id] = len(data) - 1

            # Max number of entities filter.
            if len(data) == max_number:
                break

        if do_sort_by_duration:
            if index_by_file_id:
                logging.warning("Tried to sort dataset by duration, but cannot since index_by_file_id is set.")
            else:
                data.sort(key=lambda entity: entity.duration)

        logging.info(
            "Filtered duration for loading collection is %f.", duration_filtered,
        )
        self.uniq_labels = sorted(set(map(lambda x: x.label, data)))
        logging.info("# {} files loaded accounting to # {} labels".format(len(data), len(self.uniq_labels)))

        super().__init__(data)


class ASRSpeechLabel(SpeechLabel):
    """`SpeechLabel` collector from structured json files."""

    def __init__(self, manifests_files: Union[str, List[str]], is_regression_task=False, *args, **kwargs):
        """Parse lists of audio files, durations and transcripts texts.

        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            is_regression_task: It's a regression task
            *args: Args to pass to `SpeechLabel` constructor.
            **kwargs: Kwargs to pass to `SpeechLabel` constructor.
        """
        audio_files, durations, labels, offsets = [], [], [], []

        for item in manifest.item_iter(manifests_files, parse_func=self.__parse_item):
            audio_files.append(item['audio_file'])
            durations.append(item['duration'])
            if not is_regression_task:
                labels.append(item['label'])
            else:
                labels.append(float(item['label']))

            offsets.append(item['offset'])

        super().__init__(audio_files, durations, labels, offsets, *args, **kwargs)

    def __parse_item(self, line: str, manifest_file: str) -> Dict[str, Any]:
        item = json.loads(line)

        # Audio file
        if 'audio_filename' in item:
            item['audio_file'] = item.pop('audio_filename')
        elif 'audio_filepath' in item:
            item['audio_file'] = item.pop('audio_filepath')
        else:
            raise ValueError(
                f"Manifest file has invalid json line " f"structure: {line} without proper audio file key."
            )
        item['audio_file'] = os.path.expanduser(item['audio_file'])

        # Duration.
        if 'duration' not in item:
            raise ValueError(f"Manifest file has invalid json line " f"structure: {line} without proper duration key.")

        # Label.
        if 'command' in item:
            item['label'] = item.pop('command')
        elif 'target' in item:
            item['label'] = item.pop('target')
        elif 'label' in item:
            pass
        else:
            raise ValueError(f"Manifest file has invalid json line " f"structure: {line} without proper label key.")

        item = dict(
            audio_file=item['audio_file'],
            duration=item['duration'],
            label=item['label'],
            offset=item.get('offset', None),
        )

        return item


class FeatureSequenceLabel(_Collection):
    """List of feature sequence of label correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(typename='FeatureSequenceLabelEntity', field_names='feature_file seq_label',)

    def __init__(
        self,
        feature_files: List[str],
        seq_labels: List[str],
        max_number: Optional[int] = None,
        index_by_file_id: bool = False,
    ):
        """Instantiates feature-SequenceLabel manifest with filters and preprocessing.

        Args:
            feature_files: List of feature files.
            seq_labels: List of sequences of abels.
            max_number: Maximum number of samples to collect.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
        """

        output_type = self.OUTPUT_TYPE
        data, num_filtered = (
            [],
            0.0,
        )
        self.uniq_labels = set()

        if index_by_file_id:
            self.mapping = {}

        for feature_file, seq_label in zip(feature_files, seq_labels):

            label_tokens, uniq_labels_in_seq = self.relative_speaker_parser(seq_label)

            data.append(output_type(feature_file, label_tokens))
            self.uniq_labels |= uniq_labels_in_seq

            if label_tokens is None:
                num_filtered += 1
                continue

            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(feature_file))
                self.mapping[feature_file] = len(data) - 1

            # Max number of entities filter.
            if len(data) == max_number:
                break

        logging.info("# {} files loaded including # {} unique labels".format(len(data), len(self.uniq_labels)))
        super().__init__(data)

    def relative_speaker_parser(self, seq_label):
        """ Convert sequence of speaker labels to relative labels.
        Convert sequence of absolute speaker to sequence of relative speaker [E A C A E E C] -> [0 1 2 1 0 0 2]
        In this seq of label , if label do not appear before, assign new relative labels len(pos); else reuse previous assigned relative labels.
        Args:
            seq_label (str): A string of a sequence of labels.

        Return:
            relative_seq_label (List) : A list of relative sequence of labels
            unique_labels_in_seq (Set): A set of unique labels in the sequence
        """
        seq = seq_label.split()
        conversion_dict = dict()
        relative_seq_label = []

        for seg in seq:
            if seg in conversion_dict:
                converted = conversion_dict[seg]
            else:
                converted = len(conversion_dict)
                conversion_dict[seg] = converted

            relative_seq_label.append(converted)

        unique_labels_in_seq = set(conversion_dict.keys())
        return relative_seq_label, unique_labels_in_seq


class ASRFeatureSequenceLabel(FeatureSequenceLabel):
    """`FeatureSequenceLabel` collector from asr structured json files."""

    def __init__(
        self, manifests_files: Union[str, List[str]], max_number: Optional[int] = None, index_by_file_id: bool = False,
    ):

        """Parse lists of feature files and sequences of labels.

        Args:
            manifests_files: Either single string file or list of such -
                manifests to yield items from.
            max_number:  Maximum number of samples to collect; pass to `FeatureSequenceLabel` constructor.
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data; pass to `FeatureSequenceLabel` constructor.
        """

        feature_files, seq_labels = [], []
        for item in manifest.item_iter(manifests_files, parse_func=self._parse_item):
            feature_files.append(item['feature_file'])
            seq_labels.append(item['seq_label'])

        super().__init__(feature_files, seq_labels, max_number, index_by_file_id)

    def _parse_item(self, line: str, manifest_file: str) -> Dict[str, Any]:
        item = json.loads(line)

        # Feature file
        if 'feature_filename' in item:
            item['feature_file'] = item.pop('feature_filename')
        elif 'feature_filepath' in item:
            item['feature_file'] = item.pop('feature_filepath')
        else:
            raise ValueError(
                f"Manifest file has invalid json line " f"structure: {line} without proper feature file key."
            )
        item['feature_file'] = os.path.expanduser(item['feature_file'])

        # Seq of Label.
        if 'seq_label' in item:
            item['seq_label'] = item.pop('seq_label')
        else:
            raise ValueError(
                f"Manifest file has invalid json line " f"structure: {line} without proper seq_label key."
            )

        item = dict(feature_file=item['feature_file'], seq_label=item['seq_label'],)

        return item

class DiarizationLabel(_Collection):
    """List of diarization audio-label correspondence with preprocessing."""

    OUTPUT_TYPE = collections.namedtuple(
        typename='DiarizationLabelEntity',
        field_names='audio_file duration rttm_file offset target_spks sess_spk_dict clus_spk_digits rttm_spk_digits',
    )

    def __init__(
        self,
        audio_files: List[str],
        durations: List[float],
        rttm_files: List[str],
        offsets: List[float],
        target_spks_list: List[tuple],
        sess_spk_dicts: List[Dict],
        clus_spk_list: List[tuple],
        rttm_spk_list: List[tuple],
        max_number: Optional[int] = None,
        do_sort_by_duration: bool = False,
        index_by_file_id: bool = False,
    ):
        """Instantiates audio-label manifest with filters and preprocessing.

        Args:
            audio_files:
                List of audio file paths.
            durations:
                List of float durations.
            rttm_files:
                List of RTTM files (Groundtruth diarization annotation file).
            offsets:
                List of offsets or None.
            target_spks (tuple):
                List of tuples containing the two indices of targeted speakers for evaluation.
                Example: [[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], [(0, 1), (1, 2), (0, 2)], ...]
            sess_spk_dict (Dict):
                List of Mapping dictionaries between RTTM speakers and speaker labels in the clustering result.
            clus_spk_digits (tuple):
                List of Tuple containing all the speaker indices from the clustering result.
                Example: [(0, 1, 2, 3), (0, 1, 2), ...]
            rttm_spkr_digits (tuple):
                List of tuple containing all the speaker indices in the RTTM file.
                Example: (0, 1, 2), (0, 1), ...]
            max_number: Maximum number of samples to collect
            do_sort_by_duration: True if sort samples list by duration
            index_by_file_id: If True, saves a mapping from filename base (ID) to index in data.
        """

        if index_by_file_id:
            self.mapping = {}
        output_type = self.OUTPUT_TYPE
        data, duration_filtered = [], 0.0

        zipped_items = zip(
            audio_files, durations, rttm_files, offsets, target_spks_list, sess_spk_dicts, clus_spk_list, rttm_spk_list
        )
        for (
            audio_file,
            duration,
            rttm_file,
            offset,
            target_spks,
            sess_spk_dict,
            clus_spk_digits,
            rttm_spk_digits,
        ) in zipped_items:

            if duration is None:
                duration = 0

            data.append(
                output_type(
                    audio_file,
                    duration,
                    rttm_file,
                    offset,
                    target_spks,
                    sess_spk_dict,
                    clus_spk_digits,
                    rttm_spk_digits,
                )
            )

            if index_by_file_id:
                file_id, _ = os.path.splitext(os.path.basename(audio_file))
                self.mapping[file_id] = len(data) - 1

            # Max number of entities filter.
            if len(data) == max_number:
                break

        if do_sort_by_duration:
            if index_by_file_id:
                logging.warning("Tried to sort dataset by duration, but cannot since index_by_file_id is set.")
            else:
                data.sort(key=lambda entity: entity.duration)

        logging.info(
            "Filtered duration for loading collection is %f.", duration_filtered,
        )
        logging.info(f"Total {len(data)} session files loaded accounting to # {len(audio_files)} audio clips")

        super().__init__(data)


class DiarizationSpeechLabel(DiarizationLabel):
    """`DiarizationLabel` diarization data sample collector from structured json files."""

    def __init__(
        self,
        manifests_files: Union[str, List[str]],
        emb_dict: Dict,
        clus_label_dict: Dict,
        round_digit=2,
        seq_eval_mode=False,
        pairwise_infer=False,
        *args,
        **kwargs,
    ):
        """
        Parse lists of audio files, durations, RTTM (Diarization annotation) files. Since diarization model infers only
        two speakers, speaker pairs are generated from the total number of speakers in the session.

        Args:
            manifest_filepath (str):
                 Path to input manifest json files.
            emb_dict (Dict):
                Dictionary containing cluster-average embeddings and speaker mapping information.
            clus_label_dict (Dict):
                Segment-level speaker labels from clustering results.
            round_digit (int):
                Number of digits to be rounded.
            seq_eval_mode (bool):
                If True, F1 score will be calculated for each speaker pair during inference mode.
            pairwise_infer (bool):
                If True, this Dataset class operates in inference mode. In inference mode, a set of speakers in the input audio
                is split into multiple pairs of speakers and speaker tuples (e.g. 3 speakers: [(0,1), (1,2), (0,2)]) and then
                fed into the diarization system to merge the individual results.
            *args: Args to pass to `SpeechLabel` constructor.
            **kwargs: Kwargs to pass to `SpeechLabel` constructor.
        """
        self.round_digit = round_digit
        self.emb_dict = emb_dict
        self.clus_label_dict = clus_label_dict
        self.seq_eval_mode = seq_eval_mode
        self.pairwise_infer = pairwise_infer
        audio_files, durations, rttm_files, offsets, target_spks_list, sess_spk_dicts, clus_spk_list, rttm_spk_list = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for item in manifest.item_iter(manifests_files, parse_func=self.__parse_item_rttm):
            # Inference mode
            if self.pairwise_infer:
                clus_speaker_digits = sorted(list(set([x[2] for x in clus_label_dict[item['uniq_id']]])))
                if item['rttm_file']:
                    base_scale_index = max(self.emb_dict.keys())
                    _sess_spk_dict = self.emb_dict[base_scale_index][item['uniq_id']]['mapping']
                    sess_spk_dict = {int(v.split('_')[-1]): k for k, v in _sess_spk_dict.items()}
                    rttm_speaker_digits = [int(v.split('_')[1]) for k, v in _sess_spk_dict.items()]
                    if self.seq_eval_mode:
                        clus_speaker_digits = rttm_speaker_digits
                else:
                    sess_spk_dict = None
                    rttm_speaker_digits = None
            # Training mode
            else:
                sess_spk_dict = get_rttm_speaker_index(rttm_to_labels(item['rttm_file']))
                target_spks = tuple(sess_spk_dict.keys())
                clus_speaker_digits = target_spks
                rttm_speaker_digits = target_spks

            if len(clus_speaker_digits) <= 2:
                spk_comb_list = [(0, 1)]
            else:
                spk_comb_list = [x for x in combinations(clus_speaker_digits, 2)]

            for target_spks in spk_comb_list:
                audio_files.append(item['audio_file'])
                durations.append(item['duration'])
                rttm_files.append(item['rttm_file'])
                offsets.append(item['offset'])
                target_spks_list.append(target_spks)
                sess_spk_dicts.append(sess_spk_dict)
                clus_spk_list.append(clus_speaker_digits)
                rttm_spk_list.append(rttm_speaker_digits)

        super().__init__(
            audio_files,
            durations,
            rttm_files,
            offsets,
            target_spks_list,
            sess_spk_dicts,
            clus_spk_list,
            rttm_spk_list,
            *args,
            **kwargs,
        )

    def __parse_item_rttm(self, line: str, manifest_file: str) -> Dict[str, Any]:
        """Parse each rttm file and save it to in Dict format"""
        item = json.loads(line)
        if 'audio_filename' in item:
            item['audio_file'] = item.pop('audio_filename')
        elif 'audio_filepath' in item:
            item['audio_file'] = item.pop('audio_filepath')
        else:
            raise ValueError(
                f"Manifest file has invalid json line " f"structure: {line} without proper audio file key."
            )
        item['audio_file'] = os.path.expanduser(item['audio_file'])
        item['uniq_id'] = os.path.splitext(os.path.basename(item['rttm_filepath']))[0]
        if 'duration' not in item:
            raise ValueError(f"Manifest file has invalid json line " f"structure: {line} without proper duration key.")
        item = dict(
            audio_file=item['audio_file'],
            uniq_id=item['uniq_id'],
            duration=item['duration'],
            rttm_file=item['rttm_filepath'],
            offset=item.get('offset', None),
        )
        return item


class SyntheticDiarizationSpeechLabel(_Collection):
    """Synthetic Diarization Session Generator."""

    OUTPUT_TYPE = collections.namedtuple(
        typename='DiarizationLabelEntity',
        field_names='audio_file duration rttm_file offset target_spks sess_spk_dict clus_spk_digits rttm_spk_digits',
    )

    def __init__(
        self,
        cfg_path: str,
        pairwise_infer: bool,
    ):
        """
            Instantiates synthetic diarization session generator.
        """
        with open(cfg_path, 'r') as f:
            self._params = yaml.safe_load(f)
        self.data_simulator = LibriSpeechGenerator(self._params) #includes tmp dir
        self.pairwise_infer = pairwise_infer

        self._sample_counter = 0
        self.collection = None

    def create_base_manifest(self, wav_path, text_path=None, rttm_path=None, uem_path=None, ctm_path=None, manifest_filepath=None):
        if os.path.exists(manifest_filepath):
            os.remove(manifest_filepath)
        wav_pathlist = read_file(wav_path)
        wav_pathdict = get_dict_from_wavlist(wav_pathlist)
        len_wavs = len(wav_pathlist)
        uniqids = sorted(wav_pathdict.keys())

        text_pathdict = get_path_dict(text_path, uniqids, len_wavs)
        rttm_pathdict = get_path_dict(rttm_path, uniqids, len_wavs)
        uem_pathdict = get_path_dict(uem_path, uniqids, len_wavs)
        ctm_pathdict = get_path_dict(ctm_path, uniqids, len_wavs)

        lines = []
        for uid in uniqids:
            wav, text, rttm, uem, ctm = (
                wav_pathdict[uid],
                text_pathdict[uid],
                rttm_pathdict[uid],
                uem_pathdict[uid],
                ctm_pathdict[uid],
            )

            audio_line = wav.strip()
            if rttm is not None:
                rttm = rttm.strip()
                labels = rttm_to_labels(rttm)
                num_speakers = Counter([l.split()[-1] for l in labels]).keys().__len__()
            else:
                num_speakers = None

            if uem is not None:
                uem = uem.strip()

            if text is not None:
                text = open(text.strip()).readlines()[0].strip()
            else:
                text = "-"

            if ctm is not None:
                ctm = ctm.strip()

            meta = [
                {
                    "audio_filepath": audio_line,
                    "offset": 0,
                    "duration": None,
                    "label": "infer",
                    "text": text,
                    "num_speakers": num_speakers,
                    "rttm_filepath": rttm,
                    "uem_filepath": uem,
                    "ctm_filepath": ctm,
                }
            ]
            lines.extend(meta)

        write_file(manifest_filepath, lines, range(len(lines)))

    def create_segment_manifest(self, input_manifest_path, output_manifest_path, window, shift, step_count, deci):
        if '.json' not in input_manifest_path:
            raise ValueError("input_manifest_path file should be .json file format")
        if output_manifest_path and '.json' not in output_manifest_path:
            raise ValueError("output_manifest_path file should be .json file format")
        elif not output_manifest_path:
            output_manifest_path = rreplace(input_manifest_path, '.json', f'_{step_count}seg.json')

        input_manifest_dict = get_input_manifest_dict(input_manifest_path)
        segment_manifest_path = rreplace(input_manifest_path, '.json', '_seg.json')
        subsegment_manifest_path = rreplace(input_manifest_path, '.json', '_subseg.json')
        min_subsegment_duration=0.05
        step_count = int(step_count)

        input_manifest_file = open(input_manifest_path, 'r').readlines()
        input_manifest_file = sorted(input_manifest_file)
        AUDIO_RTTM_MAP = audio_rttm_map(input_manifest_path)
        segments_manifest_file = write_rttm2manifest(AUDIO_RTTM_MAP, segment_manifest_path, deci)
        print(segments_manifest_file)
        subsegments_manifest_file = subsegment_manifest_path
        segments_manifest_to_subsegments_manifest(
            segments_manifest_file,
            subsegments_manifest_file,
            window,
            shift,
            min_subsegment_duration,
        )
        subsegments_dict = get_subsegment_dict(subsegments_manifest_file, window, shift, deci)
        write_truncated_subsegments(input_manifest_dict, subsegments_dict, output_manifest_path, step_count, deci)
        os.remove(segment_manifest_path)
        os.remove(subsegment_manifest_path)

    def __getitem__(self, index):
        if self._sample_counter % self._params.data_simulator.samples_per_refresh:
            #generate sessions
            self.data_simulator.generate_session()

            #update manifest_files using tmp dir
            basepath = self._params.data_simulator.output_dir
            filename = self._params.data_simulator.output_filename
            wav_path = os.path.join(basepath, 'synthetic_wav.list')
            text_path = os.path.join(basepath, 'synthetic_txt.list')
            rttm_path = os.path.join(basepath, 'synthetic_rttm.list')
            ctm_path = os.path.join(basepath, 'synthetic_ctm.list')
            manifest_filepath = os.path.join(basepath, 'base_manifest.json')
            base_manifest = self.create_base_manifest(wav_path, text_path=text_path, rttm_path=rttm_path, ctm_path=ctm_path, manifest_filepath=manifest_filepath)

            #create segmented manifest
            segment_manifest = os.path.join(basepath, 'segment_manifest.json')
            window = 0.5
            shift = 0.25
            step_count = 50
            deci = 3
            self.create_segment_manifest(base_manifest, segment_manifest, window, shift, step_count, deci)

            #reresh diarization session manifest
            self.collection = DiarizationSpeechLabel(
                manifests_files=segment_manifest,
                emb_dict=None,
                clus_label_dict=None,
                pairwise_infer=pairwise_infer,
            )

            self._sample_counter = 1
        else:
            self._sample_counter += 1

        sample = self.collection[index]

        return sample
