#!/bin/bash

# Create manifest file with alignments
python create_alignment_manifest.py --input_manifest_filepath /home/chooper/projects/datasets/LibriSpeech/dev_clean.json --base_alignment_path /home/chooper/projects/branch_nemo/NeMo/scripts/speaker_tasks/LibriSpeech_Alignments/ --dataset dev-clean --output_path dev-clean-align.json
# Create diarization session
python create_diarization_dataset_librispeech.py --input_manifest_filepath ./dev-clean-align.json --output_dir outputs --num_speakers 3 --session_length 300 --num_sessions 1
