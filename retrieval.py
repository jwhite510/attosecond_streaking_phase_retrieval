import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy import interpolate

with open('./experimental_data/53asstreakingdata.csv') as csvfile:

    reader = csv.reader(csvfile)
    matrix = np.array(list(reader))

    Energy = matrix[4:, 0].astype('float')
    Delay = matrix[2, 2:].astype('float')
    Values = matrix[4:, 2:].astype('float')



# map the function onto a
interp2 = interpolate.interp2d(Delay, Energy, Values, kind='cubic')

delay_new = np.linspace(Delay[0], Delay[-1], 176)
energy_new = np.linspace(Energy[0], Energy[-1], 200)

values_new = interp2(delay_new, energy_new)


fig = plt.figure()
gs = fig.add_gridspec(2, 2)

ax = fig.add_subplot(gs[0, :])
ax.pcolormesh(Delay, Energy, Values, cmap='jet')
ax.set_xlabel('fs')
ax.set_ylabel('eV')
ax.text(0.1, 0.9, 'original', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[1, :])
ax.pcolormesh(values_new, cmap='jet')
ax.set_xlabel('fs')
ax.set_ylabel('eV')
ax.text(0.1, 0.9, 'interpolated', transform=ax.transAxes, backgroundcolor='white')



plt.show()
