import numpy as np
import scipy.constants as sc


# includes linear
xuv_phase_coefs=5
# phase amplitude
amplitude=8.0
# amplitude=4.0

#infrared params
ir_param_amplitudes = {}
ir_param_amplitudes["phase_range"] = (0, 2 * np.pi)
ir_param_amplitudes["clambda_range"] = (1.6345, 1.6345)
# ir_param_amplitudes["clambda_range"] = (1.0, 1.6345)
ir_param_amplitudes["pulseduration_range"] =  (7.0, 12.0)
ir_param_amplitudes["I_range"] = (0.4, 1.0)




#---------------------------
#--STREAKING TRACE PARAMS---
#---------------------------
Ip_eV = 21.5645
Ip = Ip_eV * sc.electron_volt  # joules
Ip = Ip / sc.physical_constants['atomic unit of energy'][0]  # a.u.

# define delay values
# these must be smaller values than the IR pulse window timespan (a.u.)
delay_values = np.array([1, 2, 3])









# threshold scaler for the generated pulses
threshold_scaler = 0.03

# threshold_min_index = 100
threshold_min_index = 50
# threshold_max_index = (2*1024) - 100
threshold_max_index = 1024 - 50







