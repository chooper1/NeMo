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

random.seed(42)

"""
This script creates a room impulse response using the gpuRIR library and then
simulates the trajectory for a selected audio source in this room.
"""


def main():
    input_audio_filepath = args.input_audio_filepath
    output_path = args.output_path
    # from the example: https://github.com/DavidDiazGuerra/gpuRIR/blob/master/examples/example.py
    # parameter values explained here: https://github.com/DavidDiazGuerra/gpuRIR#simulatetrajectory
    room_sz = [3, 3, 2.5]  # Size of the room [m]
    nb_src = 4  # Number of sources
    pos_src = np.array([[0.6, 1.1, 0.5], [1, 2, 0.5], [0.4, 1.1, 0.5], [1, 2.1, 0.5]])  # Positions of the sources ([m]
    nb_rcv = 2  # Number of receivers
    pos_rcv = np.array([[0.5, 1, 0.5],[1, 1, 0.5]])  # Position of the receivers [m]
    orV_rcv = None  # Vectors pointing in the same direction than the receivers (None assumes omnidirectional)
    mic_pattern = "omni"  # Receiver polar pattern
    abs_weights = [0.4] * 5 + [0.2] #[0.9] * 5 + [0.5]  # Absortion coefficient ratios of the walls
    T60 = 1  # Time for the RIR to reach 60dB of attenuation [s]
    att_diff = 15.0  # Attenuation when start using the diffuse reverberation model [dB]
    att_max = 60.0  # Attenuation at the end of the simulation [dB]
    fs = 16000.0  # Sampling frequency [Hz]

    beta = beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights)  # Reflection coefficients
    Tdiff = att2t_SabineEstimator(att_diff, T60)  # Time to start the diffuse reverberation model [s]
    Tmax = att2t_SabineEstimator(att_max, T60)  # Time to stop the simulation [s]
    nb_img = t2n(Tdiff, room_sz)  # Number of image sources in each dimension
    RIR = simulateRIR(
        room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern
    )

    print(RIR.shape)

    os.mkdir('./RIR')
    with open('./RIR/rir1.npy', 'wb') as f:
        np.save(f, RIR)

    # from https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/room.py#2216
    # need to convolve individual audio sources with separate RIRs
    input_wav, sr = librosa.load(input_audio_filepath, sr=fs)

    speaker_id = 0
    output_sound = []
    for channel in range(0,nb_rcv):
        out_channel = convolve(input_wav, RIR[channel, speaker_id, : len(input_wav)]).tolist()
        output_sound.append(out_channel)
    output_sound = np.array(output_sound).T
    output_sound = output_sound / np.max(np.abs(output_sound))  # normalize to [-1,1]
    sf.write(output_path, output_sound, int(fs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RIR Creator")
    parser.add_argument("--input_audio_filepath", help="path to input audio file", type=str, default="/home/chooper/projects/datasets/LibriSpeech/LibriSpeech/dev-clean-processed/2277-149874-0000.wav")
    parser.add_argument("--output_path", help="path to output file", type=str, default='./test/diarization_session_0.wav')
    args = parser.parse_args()
    main()
