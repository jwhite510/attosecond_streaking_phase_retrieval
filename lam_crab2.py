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
        self.gdd = 1000 * atts**2 # gdd

        #discretize
        self.tmax = 30 * self.t0
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
        self.Et = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.Et))) * (np.exp(0.5j * self.gdd * (2 * np.pi)**2) * self.fmat))))


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
fmat = np.roll(ir.fmat, int(len(ir.fmat)/2))
dtau = ir.dt


# set up the XUV time axis
nt = len(xuv.tmat)
tmat = xuv.tmat
dt = xuv.dt

# construct the XUV spectral axis
enmat = (2 * xuv.en0)/nt * np.arange(0, xuv.N, 1).reshape(1, 1, -1)

# compute the IR fields vector potential
At = -dtau * np.cumsum(ir.Et)

# Compute the integral of the driving IR field vector potential
Bt = dtau * np.cumsum(At)


# compute the phase gate
thing = np.exp(-2j * np.pi * np.transpose(np.outer(tmat, fmat)))
Ct = -Bt[-1] + np.transpose(np.real(np.fft.ifft(np.fft.fft(Bt, axis=0).reshape(-1, 1) * thing, axis=0)))

# compute the electron field
Etx = xuv.Et.reshape(-1, 1)
thing2 = (enmat + med.Ip) * tmat.reshape(-1, 1, 1)
## .................

print(np.shape(thing2))





exit(0)






# compute the electron field
xuv.Et = np.array([1, 2, 3, 4 ,5, 6, 7, 8])
Etx = np.array(xuv.Et).reshape(1, -1) * np.ones((nt, nt))

print(np.shape(Etx))
print(Etx)










