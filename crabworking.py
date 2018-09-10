import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc


# SI units for defining parameters
W = 1
cm = 1e-2
um = 1e-6
fs = 1e-15
atts = 1e-18

class XUV_Field():

    def __init__(self, N):

        # define parameters in SI units
        self.N = N
        self.en0 = 150 * sc.eV # central energy
        self.den0 = 75 * sc.eV #energy fwhm
        self.f0 = self.en0/sc.h # carrier frequency
        self.T0 = 1/self.f0 # optical cycle
        self.t0 = 2 * sc.h * np.log(2) / (np.pi * self.den0) # pulse duration
        self.gdd = 50 * atts**2 # gdd
        self.gdd_si = self.gdd / atts**2

        #discretize
        self.tmax = 25 * self.t0
        self.dt = self.tmax / N
        self.tmat = self.dt * np.arange(-N/2, N/2, 1)

        # discretize the streaking xuv field spectral matrix
        self.df = 1/(self.dt * N)
        self.fmat = self.df * np.arange(-N/2, N/2, 1)
        self.enmat = sc.h * self.fmat

        # convert to AU
        self.en0 = self.en0 / sc.physical_constants['atomic unit of energy'][0]
        self.den0 = self.den0 / sc.physical_constants['atomic unit of energy'][0]
        self.t0 = self.t0 / sc.physical_constants['atomic unit of time'][0]
        self.f0 = self.f0 * sc.physical_constants['atomic unit of time'][0]
        self.T0 = self.T0 / sc.physical_constants['atomic unit of time'][0]
        self.gdd = self.gdd / sc.physical_constants['atomic unit of time'][0]**2
        self.dt = self.dt / sc.physical_constants['atomic unit of time'][0]
        self.tmat = self.tmat / sc.physical_constants['atomic unit of time'][0]
        self.fmat = self.fmat * sc.physical_constants['atomic unit of time'][0]
        self.enmat = self.enmat / sc.physical_constants['atomic unit of energy'][0]

        # set up streaking xuv field in AU
        self.Et = np.exp(-2 * np.log(2) * (self.tmat/self.t0)**2 ) * np.exp(2j * np.pi * self.f0 * self.tmat)

        # add GDD to streaking XUV field
        Ef = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.Et)))
        Ef_prop = Ef * np.exp(0.5j * self.gdd * (2 * np.pi)**2 * (self.fmat - self.f0)**2)
        self.Et = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ef_prop)))


class IR_Field():

    def __init__(self, N):
        self.N = N
        # calculate parameters in SI units
        self.lam0 = 1.7 * um    # central wavelength
        self.f0 = sc.c/self.lam0    # carrier frequency
        self.T0 = 1/self.f0 # optical cycle
        self.t0 = 12 * fs # pulse duration
        self.ncyc = self.t0/self.T0
        self.I0 = 1e13 * W/cm**2

        # compute ponderomotive energy
        self.Up = (sc.elementary_charge**2 * self.I0) / (2 * sc.c * sc.epsilon_0 * sc.electron_mass * (2 * np.pi * self.f0)**2)

        # discretize time matrix
        self.tmax = 8 * self.t0
        self.dt = self.tmax / N
        self.tmat = self.dt * np.arange(-N/2, N/2, 1)

        # discretize spectral matrix
        self.df = 1/(self.dt * N)
        self.fmat = self.df * np.arange(-N/2, N/2, 1)
        self.enmat = sc.h * self.fmat

        # convert units to AU
        self.t0 = self.t0 / sc.physical_constants['atomic unit of time'][0]
        self.f0 = self.f0 * sc.physical_constants['atomic unit of time'][0]
        self.T0 = self.T0 / sc.physical_constants['atomic unit of time'][0]
        self.Up = self.Up / sc.physical_constants['atomic unit of energy'][0]
        self.dt = self.dt / sc.physical_constants['atomic unit of time'][0]
        self.tmat = self.tmat / sc.physical_constants['atomic unit of time'][0]
        self.fmat = self.fmat * sc.physical_constants['atomic unit of time'][0]
        self.enmat = self.enmat / sc.physical_constants['atomic unit of energy'][0]

        # calculate driving amplitude in AU
        self.E0 = np.sqrt(4 * self.Up * (2 * np.pi * self.f0)**2)

        # set up the driving IR field amplitude in AU
        self.Et = self.E0 * np.exp(-2 * np.log(2) * (self.tmat/self.t0)**2) * np.cos(2 * np.pi * self.f0 * self.tmat)


class Med():

    def __init__(self):

        self.Ip = 24.587 * sc.electron_volt
        self.Ip = self.Ip / sc.physical_constants['atomic unit of energy'][0]



xuv = XUV_Field(N=2**9)
ir = IR_Field(N=256)
med = Med()


# set up the IR delay axis
taumat = ir.tmat
# fmat = np.roll(ir.fmat, int(len(ir.fmat)/2))
dtau = ir.dt


# set up the XUV time axis
nt = len(xuv.tmat)
tmat = xuv.tmat
dt = xuv.dt

# construct the XUV spectral axis
enmat = (2 * xuv.en0)/nt * np.arange(0, xuv.N, 1).reshape(1, 1, -1)

# construct delay axis
delaygrid = ir.fmat.reshape(1, -1) * xuv.tmat.reshape(-1, 1)

# fourier transform ir.Et
irEf = np.fft.fftshift(np.fft.fft(np.fft.fftshift(ir.Et))).reshape(1, -1)

# apply phase in frequency domain
irEf_phase_applied = irEf * np.exp(-1j * 2 * np.pi * delaygrid)

# return to temporal domain
irEt_delay = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(irEf_phase_applied)))

# construct A_t and reverse integral of A_t
A_t = - ir.dt * np.cumsum(irEt_delay, axis=1)
A_t_int = - ir.dt * np.flip(np.cumsum(np.flip(A_t, axis=1), axis=1), axis=1)

# momentum
p = np.sqrt(2 * enmat)

# calculate phase gate
phi_g = np.exp(1j * p * np.expand_dims(A_t_int, axis=2))

# XUV field
Exuv = xuv.Et.reshape(-1, 1, 1)

# fourier transform exponential
t = xuv.tmat.reshape(-1, 1, 1)
ftexp = np.exp(-1j * (enmat + med.Ip) * t)

# product
product = Exuv * ftexp * phi_g

integral = np.sum(product, axis=0)


plt.pcolormesh(np.transpose(np.abs(integral)**2))
plt.show()




exit(0)








