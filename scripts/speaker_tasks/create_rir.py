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
import numpy as np
import numpy.matlib as matlib

from gpuRIR import simulateRIR #use simulateTrajectory for moving sources

random.seed(42)

"""
This script creates a room impulse response using the gpuRIR library and then
simulates the trajectory for a selected audio source in this room.
"""

def main():
    # from the example: https://github.com/DavidDiazGuerra/gpuRIR/blob/master/examples/example.py
    room_sz = [3,3,2.5]  # Size of the room [m]
    nb_src = 2  # Number of sources
    pos_src = np.array([[1,2.9,0.5],[1,2,0.5]]) # Positions of the sources ([m]
    nb_rcv = 3 # Number of receivers
    pos_rcv = np.array([[0.5,1,0.5],[1,1,0.5],[1.5,1,0.5]])	 # Position of the receivers [m]
    orV_rcv = matlib.repmat(np.array([0,1,0]), nb_rcv, 1) # Vectors pointing in the same direction than the receivers
    mic_pattern = "card" # Receiver polar pattern
    abs_weights = [0.9]*5+[0.5] # Absortion coefficient ratios of the walls
    T60 = 1.0	 # Time for the RIR to reach 60dB of attenuation [s]
    att_diff = 15.0	# Attenuation when start using the diffuse reverberation model [dB]
    att_max = 60.0 # Attenuation at the end of the simulation [dB]
    fs=16000.0 # Sampling frequency [Hz]

    beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights) # Reflection coefficients
    Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60) # Time to start the diffuse reverberation model [s]
    Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)	 # Time to stop the simulation [s]
    nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension
    RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="RIR Generator")
    # parser.add_argument("--input_manifest_filepath", help="path to input manifest file", type=str, required=True)
    # args = parser.parse_args()

    main()