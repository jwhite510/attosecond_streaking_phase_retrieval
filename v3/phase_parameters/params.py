import numpy as np
import scipy.constants as sc
import pickle
import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.join(current_path+".."))
import measured_trace.get_trace as measured_trace


# includes linear
xuv_phase_coefs=5
# phase amplitude
amplitude=20.0
# amplitude=4.0

#infrared params
# for sample 2
# ir_param_amplitudes = {}
# ir_param_amplitudes["phase_range"] = (0, 2 * np.pi)
# ir_param_amplitudes["clambda_range"] = (1.6345, 1.6345)
# # ir_param_amplitudes["clambda_range"] = (1.0, 1.6345)
# ir_param_amplitudes["pulseduration_range"] =  (7.0, 12.0)
# ir_param_amplitudes["I_range"] = (0.4, 1.0)


# for sample 3
ir_param_amplitudes = {}
ir_param_amplitudes["phase_range"] = (0, 2 * np.pi)
ir_param_amplitudes["clambda_range"] = (1.678, 1.678)
# ir_param_amplitudes["clambda_range"] = (1.0, 1.6345)
ir_param_amplitudes["pulseduration_range"] = (11.0, 16.0)
ir_param_amplitudes["I_range"] = (0.02, 0.12)



#---------------------------
#--STREAKING TRACE PARAMS---
#---------------------------

# because trace 4 is using the electron specturm, dont apply ionization potential
Ip_eV = 24.587 # eV
Ip = Ip_eV * sc.electron_volt  # joules
Ip = Ip / sc.physical_constants['atomic unit of energy'][0]  # a.u.
# sample = 2
delay_values = measured_trace.delay
# delay_values_fs = delay_values * sc.physical_constants['atomic unit of time'][0] * 1e15
delay_values_fs = delay_values * 1e15
K = measured_trace.energy

# define delay values
# these must be smaller values than the IR pulse window timespan (a.u.)
# delay_values = np.linspace(-500, 500, 60) # a.u.
# these are the delay values from sample 2
# with open("tauvals.p", "rb") as file:
#     delay_values = pickle.load(file)
#     delay_values_fs = delay_values * sc.physical_constants['atomic unit of time'][0] * 1e15
# K = np.arange(50, 351, 1) # eV



# threshold scaler for the generated pulses
threshold_scaler = 0.03

threshold_min_index = 100
# threshold_min_index = 50
threshold_max_index = (2*1024) - 100
# threshold_max_index = 1024 - 50







