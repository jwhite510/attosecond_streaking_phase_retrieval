import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import csv
import scipy.constants as sc




def get_measured_trace():
    filepath = '../experimental_data/53asstreakingdata.csv'
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        matrix = np.array(list(reader))

        Energy = matrix[4:, 0].astype('float')
        Delay = matrix[2, 2:].astype('float')
        Values = matrix[4:, 2:].astype('float')


    # map the function onto a grid matching the training data
    interp2 = interpolate.interp2d(Delay, Energy, Values, kind='linear')
    timespan = np.abs(Delay[-1]) + np.abs(Delay[0])
    delay_new = np.arange(Delay[0], Delay[-1], timespan/160)
    energy_new = np.linspace(Energy[0], Energy[-1], 200)
    values_new = interp2(delay_new, energy_new)

    # interpolate to momentum [a.u]
    energy_new_joules = energy_new * sc.electron_volt # joules
    energy_new_au = energy_new_joules / sc.physical_constants['atomic unit of energy'][0]  # a.u.
    momentum_new_au = np.sqrt(2 * energy_new_au)
    interp2_momentum = interpolate.interp2d(delay_new, momentum_new_au, values_new, kind='linear')

    # interpolate onto linear momentum axis
    N = len(momentum_new_au)
    momentum_linear = np.linspace(momentum_new_au[0], momentum_new_au[-1], N)
    values_lin_momentum = interp2_momentum(delay_new, momentum_linear)


    return delay_new, momentum_linear, values_lin_momentum



delay, momentum, values = get_measured_trace()
plt.figure(1)
plt.pcolormesh(delay, momentum, values)
print(len(delay))
print(len(momentum))
print(delay[-1])
print(delay[0])
print(momentum[-1])
print(momentum[0])

plt.show()