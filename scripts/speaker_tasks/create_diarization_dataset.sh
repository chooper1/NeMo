#!/bin/bash

# Create manifest file with alignments
python create_alignment_manifest.py \
  --input_manifest_filepath /home/chooper/projects/datasets/LibriSpeech/dev_clean.json \
  --base_alignment_path /home/chooper/projects/branch_nemo/NeMo/scripts/speaker_tasks/LibriSpeech_Alignments/ \
  --dataset dev-clean \
  --output_path dev-clean-align.json
# Create diarization session
python create_diarization_dataset_librispeech.py \
  --input_manifest_filepath ./dev-clean-align.json \
  --output_dir outputs \
  --output_filename diarization_session \
  --num_sessions 1 \
  --num_speakers 3 \
  --session_length 300 \
  --sentence_length_k 2.81 \
  --sentence_length_p 0.1 \
  --dominance_var 0.1 \
  --min_dominance 0.05 \
  --turn_prob 0.9 \
  --mean_overlap 0.08 \
  --mean_silence 0.08 \
  --overlap_prob 0.3 \
  --outputs rjc
# note: --enforce_num_speakers is store_true
