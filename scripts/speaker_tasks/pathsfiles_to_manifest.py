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
import logging
import os
import random
from collections import Counter
from collections import OrderedDict as od

from nemo.collections.asr.parts.utils.manifest_utils import create_manifest

random.seed(42)

"""
This script creates manifest file for speaker diarization inference purposes.
Useful to get manifest when you have list of audio files and optionally rttm and uem files for evaluation

Note: make sure basename for each file is unique and rttm files also has the corresponding base name for mapping
"""

def main(wav_path, text_path=None, rttm_path=None, uem_path=None, ctm_path=None, manifest_filepath=None):
    create_manifest(wav_path, manifest_filepath, text_path=text_path, rttm_path=rttm_path, uem_path=uem_path, ctm_path=ctm_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paths2audio_files", help="path to text file containing list of audio files", type=str, required=True
    )
    parser.add_argument("--paths2txt_files", help="path to text file containing list of transcription files", type=str)
    parser.add_argument("--paths2rttm_files", help="path to text file containing list of rttm files", type=str)
    parser.add_argument("--paths2uem_files", help="path to uem files", type=str)
    parser.add_argument("--paths2ctm_files", help="path to ctm files", type=str)
    parser.add_argument("--manifest_filepath", help="path to output manifest file", type=str, required=True)

    args = parser.parse_args()

    main(
        args.paths2audio_files,
        args.paths2txt_files,
        args.paths2rttm_files,
        args.paths2uem_files,
        args.paths2ctm_files,
        args.manifest_filepath,
    )