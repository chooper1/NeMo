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
import os
import random
import shutil

from nemo.collections.asr.parts.preprocessing.diarization import LibriSpeechGenerator

random.seed(42)

"""
This script creates a synthetic diarization dataset using the LibriSpeech dataset.
"""


def main():
    input_manifest_filepath = args.input_manifest_filepath
    output_dir = args.output_dir
    num_sessions = args.num_sessions
    session_length = args.session_length
    num_speakers = args.num_speakers
    output_filename = args.output_filename
    sentence_length_params = args.sentence_length_params
    dominance_dist = args.dominance_dist
    turn_prob = args.turn_prob
    mean_overlap = args.mean_overlap
    mean_silence = args.mean_silence
    overlap_prob = args.overlap_prob

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    lg = LibriSpeechGenerator(
        manifest_path=input_manifest_filepath,
        sr=16000,
        output_dir=output_dir,
        session_length=session_length,
        num_speakers=num_speakers,
        output_filename=output_filename,
        sentence_length_params=sentence_length_params,
        dominance_dist=dominance_dist,
        turn_prob=turn_prob,
        mean_overlap=mean_overlap,
        mean_silence=mean_silence,
        overlap_prob=overlap_prob,
    )

    lg.generate_session(num_sessions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LibriSpeech Synthetic Diarization Generator")
    parser.add_argument("--input_manifest_filepath", help="path to input manifest file", type=str, required=True)
    parser.add_argument("--output_dir", help="path to output directory", type=str, required=True)
    parser.add_argument(
        "--output_filename", help="filename for wav and rttm files", type=str, default='diarization_session'
    )
    parser.add_argument("--num_sessions", help="number of diarization sessions", type=int, default=1)
    parser.add_argument("--session_length", help="length of each diarization session (seconds)", type=int, default=20)
    parser.add_argument("--num_speakers", help="number of speakers", type=int, default=2)
    parser.add_argument(
        "--sentence_length_params", help="k,p for nb distribution for sentence length", type=list, default=[2.81, 0.1]
    )
    parser.add_argument("--dominance_dist", help="distribution of speaker dominance", type=str, default="random")
    parser.add_argument("--turn_prob", help="number of speakers", type=float, default=0.9)
    parser.add_argument("--mean_overlap", help="mean percentage of overlap", type=float, default=0.08)
    parser.add_argument("--mean_silence", help="mean percentage of silence", type=float, default=0.08)
    parser.add_argument("--overlap_prob", help="probability of overlap", type=float, default=0.3)
    args = parser.parse_args()

    main()
