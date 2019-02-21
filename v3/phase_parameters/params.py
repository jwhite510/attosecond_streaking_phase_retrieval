import numpy as np
import scipy.constants as sc


# includes linear
xuv_phase_coefs=5
# phase amplitude
amplitude=9.0
# amplitude=4.0

#infrared params
ir_param_amplitudes = {}
ir_param_amplitudes["phase_range"] = (0, 2 * np.pi)
# ir_param_amplitudes["clambda_range"] = (1.6345, 1.6345)
ir_param_amplitudes["clambda_range"] = (1.0, 1.6345)
ir_param_amplitudes["pulseduration_range"] =  (7.0, 12.0)
ir_param_amplitudes["I_range"] = (0.4, 1.0)



Ip_eV = 21.5645
Ip = Ip_eV * sc.electron_volt  # joules
Ip = Ip / sc.physical_constants['atomic unit of energy'][0]  # a.u.

