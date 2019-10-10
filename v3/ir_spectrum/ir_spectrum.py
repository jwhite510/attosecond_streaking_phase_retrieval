import scipy.constants as sc
import numpy as np
import phase_parameters.params




# SI units for defining parameters
W = 1
cm = 1e-2
um = 1e-6
fs = 1e-15
atts = 1e-18


# pulse params
ir = phase_parameters.params.ir_pulse
N = ir["N"]
tmax = ir["tmax"]
start_index = ir["start_index"]
end_index = ir["end_index"]


# discretize time matrix
tmax = tmax
dt = tmax / N
tmat = dt * np.arange(-N / 2, N / 2, 1)
tmat_indexes = np.arange(int(-N / 2), int(N / 2), 1)

# discretize spectral matrix
df = 1 / (dt * N)
fmat = df * np.arange(-N / 2, N / 2, 1)

tmat_fs = np.array(tmat)*(1e15)

# convert units to AU
df = df * sc.physical_constants['atomic unit of time'][0]
dt = dt / sc.physical_constants['atomic unit of time'][0]
tmat = tmat / sc.physical_constants['atomic unit of time'][0]
fmat = fmat * sc.physical_constants['atomic unit of time'][0]

fmat_cropped = fmat[start_index: end_index]

