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
central_wavelength = measured_trace.lam0*1e6
ir_param_amplitudes = {}
ir_param_amplitudes["phase_range"] = (0, 2 * np.pi)
# use the central IR wavelength from trace
ir_param_amplitudes["clambda_range"] = (central_wavelength, central_wavelength)
# ir_param_amplitudes["clambda_range"] = (1.0, 1.6345)
ir_param_amplitudes["pulseduration_range"] = (11.0, 16.0)
ir_param_amplitudes["I_range"] = (0.02, 0.12)


#---------------------------
#--STREAKING TRACE PARAMS---
#---------------------------
Ip_eV = 24.587 # eV
Ip = Ip_eV * sc.electron_volt  # joules
Ip = Ip / sc.physical_constants['atomic unit of energy'][0]  # a.u.
delay_values = measured_trace.delay # femtosecond
delay_values_fs = delay_values * 1e15
K = measured_trace.energy

# threshold scaler for the generated pulses
threshold_scaler = 0.03

threshold_min_index = 100
# threshold_min_index = 50
threshold_max_index = (2*1024) - 100
# threshold_max_index = 1024 - 50







