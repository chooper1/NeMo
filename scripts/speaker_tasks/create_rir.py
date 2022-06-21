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

import librosa
import numpy as np
import numpy.matlib as matlib
import soundfile as sf
from gpuRIR import att2t_SabineEstimator, beta_SabineEstimation, simulateRIR, t2n
from scipy.signal import convolve  # note that scipy automatically uses fftconvolve if it is faster

from nemo.core.config import hydra_runner

random.seed(42)

"""
This script creates a room impulse response using the gpuRIR library and then
simulates the trajectory for a selected audio source in this room.
"""

@hydra_runner(config_path="conf", config_name="rir_generation.yaml")
def main(cfg):
    # from the example: https://github.com/DavidDiazGuerra/gpuRIR/blob/master/examples/example.py
    # parameter values explained here: https://github.com/DavidDiazGuerra/gpuRIR#simulatetrajectory
    output_path = cfg.rir_generation.output_path
    output_filename = cfg.rir_generation.output_filename
    room_sz = np.array(cfg.rir_generation.room_sz)
    nb_src = cfg.rir_generation.nb_src
    pos_src = np.array(cfg.rir_generation.pos_src)
    nb_rcv = cfg.rir_generation.nb_rcv
    pos_rcv = np.array(cfg.rir_generation.pos_rcv)
    orV_rcv = cfg.rir_generation.orV_rcv
    if orV_rcv:
        orV_rcv = np.array(orV_rcv)
    mic_pattern = cfg.rir_generation.mic_pattern
    abs_weights = cfg.rir_generation.abs_weights
    T60 = cfg.rir_generation.T60
    att_diff = cfg.rir_generation.att_diff
    att_max = cfg.rir_generation.att_max
    fs = cfg.rir_generation.fs

    beta = beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights)  # Reflection coefficients
    Tdiff = att2t_SabineEstimator(att_diff, T60)  # Time to start the diffuse reverberation model [s]
    Tmax = att2t_SabineEstimator(att_max, T60)  # Time to stop the simulation [s]
    nb_img = t2n(Tdiff, room_sz)  # Number of image sources in each dimension
    RIR = simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)

    if not os.path.exists(output_path):
      os.makedirs(output_path)

    with open(os.path.join(output_path, output_filename), 'wb') as f:
        np.save(f, RIR)

if __name__ == "__main__":
    main()
