import matplotlib.pyplot as plt
import tables
import numpy as np
import os
import sys
# current_path = os.path.dirname(__file__)
# sys.path.append(os.path.join(current_path+".."))
import phase_parameters.params as phase_params

N_delay = len(phase_params.delay_values_fs)
N_k = len(phase_params.K)


# open a trace from the validation set
with tables.open_file('test3.hdf5', mode='r') as hd5file:
    # regular trace
    index = 0
    trace = hd5file.root.noise_trace[index, :]
    trace = trace.reshape(N_k, N_delay)
    # proof trace
    # trace = hd5file.root.proof_trace_noise[index, :]


