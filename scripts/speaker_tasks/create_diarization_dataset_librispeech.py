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

from omegaconf import OmegaConf
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.collections.asr.parts.preprocessing.diarization import LibriSpeechGenerator

"""
This script creates a synthetic diarization session using the LibriSpeech dataset.


TODO add manifest args?
"""

@hydra_runner(config_path="conf", config_name="data_simulator.yaml")
def main(config_path):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    num_sessions = args.num_sessions
    random_seed = args.random_seed

    lg = LibriSpeechGenerator(cfg=cfg)
    lg.generate_session(num_sessions, random_seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LibriSpeech Synthetic Diarization Generator")
    parser.add_argument("--config_path", help="path to config file", type=str, required=True)
    parser.add_argument("--num_sessions", help="number of sessions to generate", type=int, default=1)
    parser.add_argument("--random_seed", help="random seed", type=int, default=42)
    args = parser.parse_args()

    main()
