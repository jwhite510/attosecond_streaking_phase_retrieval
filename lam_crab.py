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
        self.I0 = 2e12 * W/cm**2

        # compute ponderomotive energy
        self.Up = (np.e**2 * self.I0) / (2 * sc.c * sc.epsilon_0 * sc.electron_mass * (2 * np.pi * self.f0)**2)

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


xuv = XUV_Field(N=128)
ir = IR_Field(N=128)


tau = ir.tmat

# construct energy axis
e_vec = (2 * xuv.en0)/xuv.N * np.arange(0, xuv.N, 1)

tau_m, e_m, t_m = np.meshgrid(ir.tmat, e_vec, xuv.tmat)
E_xuv_m = np.ones_like(t_m) * xuv.Et.reshape(1, 1, -1)

# construct 2-d grid for delay and momentum from IR pulse
e_vec_2d, tau_m_2d= np.meshgrid(e_vec, ir.tmat)
E_ir_m = np.ones_like(tau_m_2d) * ir.Et.reshape(-1, 1)

# construct A integrals
A_tau = -1*ir.dt*np.cumsum(E_ir_m, 0)
A_tau_reverse_int = ir.dt * np.flip(np.cumsum(np.flip(A_tau, 0), 0), 0)
A_tau_reverse_int_square = ir.dt * np.flip(np.cumsum(np.flip(A_tau**2, 0), 0), 0)


# construct 3d exp(-i phi(p, t))
phi_p_t = np.sqrt(2 * e_vec_2d) * A_tau_reverse_int + 0.5 * A_tau_reverse_int_square

print('min: ', np.min(phi_p_t))
print('max: ', np.max(phi_p_t))
plt.figure(99)
plt.pcolormesh(np.sqrt(2 * e_vec_2d), cmap='jet')
plt.figure(102)
plt.pcolormesh(A_tau_reverse_int + 0.5 * A_tau_reverse_int_square, cmap='jet')

phi_p_t_exp = np.exp(-1j * phi_p_t)

plt.figure(100)
plt.pcolormesh(np.real(phi_p_t_exp), cmap='jet')


phi_p_t_3d = np.array([phi_p_t_exp]).swapaxes(0, 2) * np.ones_like(t_m)

plt.figure(5)
print(np.shape(A_tau_reverse_int))
plt.plot(A_tau_reverse_int[:, 0])
plt.plot(A_tau_reverse_int[:, 5])

e_ft = np.exp(-1j * e_m * t_m)


product = E_xuv_m * phi_p_t_3d * e_ft
product = E_xuv_m * e_ft

integrated = xuv.dt * np.sum(product, 2)
S = np.abs(integrated)**2

plt.figure(2)
plt.pcolormesh(tau, e_vec, S, cmap='jet')
plt.show()







