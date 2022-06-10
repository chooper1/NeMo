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

random.seed(42)

"""
This script creates a manifest file containing word alignments.

The alignments are obtained from: https://github.com/CorentinJ/librispeech-alignments
"""

#from scripts/speaker_tasks/filelist_to_manifest.py - move function?
def read_manifest(manifest):
    data = []
    with open(manifest, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

def main():
    input_manifest_filepath = args.input_manifest_filepath
    base_alignment_path = args.base_alignment_path
    output_dir = args.output_dir

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    manifest = read_manifest(input_manifest_filepath)

    #https://github.com/CorentinJ/librispeech-alignments/blob/master/parser_example.py
    i = 0
    while i < len(manifest):
        file = manifest[i]
        fn = file['audio_filepath'].split('/')[-1]
        speaker_id = fn.split('-')[0]
        book_id = fn.split('-')[1]
        line_id = fn.split('-')[2]

        book_dir = os.path.join(base_alignment_path, speaker_id, book_id)
        alignment_fpath = os.path.join(book_dir, f"{speaker_id}-{book_id}.alignment.txt")

        if not os.path.exists(alignment_fpath):
            raise Exception("Alignment file not found.")

        # Parse each utterance present in the file
        alignment_file = open(alignment_fpath, "r")
        for line in alignment_file:
            # Retrieve the utterance id, the words as a list and the end_times as a list
            utterance_id, words, end_times = line.strip().split(' ')
            words = words.replace('\"', '').split(',')
            end_times = [float(e) for e in end_times.replace('\"', '').split(',')]
            manifest[i,'words'] = words
            manifest[i,'alignments'] = alignments
            i+=1
        alignment_file.close()

    print(manifest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LibriSpeech Synthetic Diarization Generator")
    parser.add_argument("--input_manifest_filepath", help="path to input manifest file", type=str, required=True)
    parser.add_argument("--base_alignment_path", help="path to librispeech alignment dataset", type=str, required=True)
    parser.add_argument("--output_dir", help="path to output directory", type=str, required=True)
    args = parser.parse_args()

    main()
