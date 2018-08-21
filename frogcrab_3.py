import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as sc

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


def plot_t_cut(indexes, axis, products, number, product, axis_p, tau, p, S):
    #index[p, tau, time]

    axis[0][number].plot(np.real(product[indexes[0], indexes[1], :]), color='blue')
    axis[0][number].plot(np.imag(product[indexes[0], indexes[1], :]), color='red')

    # integrate and list number
    integral = np.abs(xuv.dt * np.sum(product[indexes[0], indexes[1], :]))**2
    axis[0][number].text(0.1, 1.1, 'integral: {}'.format(integral), transform=axis[0][number].transAxes)

    axis[1][number].plot(products['E_xuv_m'][indexes[0], indexes[1], :], color='blue', alpha=0.5, label='E_xuv_m')
    axis[1][number].plot(np.real(products['phi_p_t_3d'][indexes[0], indexes[1], :]), color='orange', alpha=0.5, label='phi_p_t_3d')
    axis[1][number].plot(np.imag(products['phi_p_t_3d'][indexes[0], indexes[1], :]), color='orange', linestyle='dashed', alpha=0.5)
    axis[1][number].plot(np.real(products['e_ft'][indexes[0], indexes[1], :]), color='red', alpha=0.1, label='e_ft')
    axis[1][number].plot(np.imag(products['e_ft'][indexes[0], indexes[1], :]), color='red', linestyle='dashed', alpha=0.1)
    if number == 3:
        axis[1][number].legend(bbox_to_anchor=(1, 0.8))

    # label the plots with corresponding numbers in the streaking trace and time axes traces
    for vertplot in [0, 1]:
        text = axis[vertplot][number].text(0.1, 0.9, number, transform=axis[vertplot][number].transAxes)
        text.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='black'))

    text = axis_p.text(tau[indexes[1]], p[indexes[0]], number, backgroundcolor='white', alpha=0.5)
    text.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='black'))
    S[indexes[0], indexes[1]] = 0


def attosecond_streak(xuv, ir, plot):

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

    phi_p_t_exp = np.exp(-1j * phi_p_t)

    phi_p_t_3d = np.array([phi_p_t_exp]).swapaxes(0, 2) * np.ones_like(t_m)


    # construct exp(i (p^2/2 + I_p)t)
    # e_ft = np.exp(1j * ((p_m**2)/2) * t_m)
    e_ft = np.exp(-1j * e_m * t_m)

    product = E_xuv_m * phi_p_t_3d * e_ft
    # product = E_xuv_m * e_ft

    fig, ax = plt.subplots(2, 3)

    ax[0][0].plot(np.real((E_xuv_m * e_ft)[0, 0, :]), color='blue')
    ax[0][1].plot(np.real(phi_p_t_3d[0, 0, :]), color='blue')
    ax[0][2].plot(np.real(product[0, 0, :]), color='blue')

    ax[1][0].plot(np.imag((E_xuv_m * e_ft)[0, 0, :]), color='red')
    ax[1][1].plot(np.imag(phi_p_t_3d[0, 0, :]), color='red')
    ax[1][2].plot(np.imag(product[0, 0, :]), color='red')

    integrated = xuv.dt * np.sum(product, 2)
    S = np.abs(integrated)**2

    if plot:
        fig, ax = plt.subplots(2, 2, figsize=(12, 4))

        print('imag average: ', np.average(np.imag(phi_p_t_exp)))
        print('real average: ', np.average(np.real(phi_p_t_exp)))

        ax[1][0].set_xlabel('tau')
        ax[1][0].set_ylabel('p')
        ax[1][0].text(0.1, 0.9, 'imag phi_p_t', transform=ax[1][0].transAxes, backgroundcolor='white')

        ax[0][1].plot(np.real(phi_p_t_exp[:, 50]))
        ax[1][1].plot(np.imag(phi_p_t_exp[:, 50]))
        # phi_p_t_exp[50, :] = 0


        ax[0][0].pcolormesh(ir.tmat, e_vec, np.transpose(np.real(phi_p_t_exp)))
        ax[0][0].set_xlabel('tau')
        ax[0][0].set_ylabel('e_vec')
        ax[0][0].text(0.1, 0.9, 'real phi_p_t', transform=ax[0][0].transAxes, backgroundcolor='white')
        ax[1][0].pcolormesh(ir.tmat, e_vec, np.transpose(np.imag(phi_p_t_exp)))





        fig, ax = plt.subplots(2, 4, figsize=(11, 5))
        products = {'E_xuv_m': E_xuv_m, 'phi_p_t_3d': phi_p_t_3d,
                    'e_ft': e_ft}

        # plot_t_cut(indexes=(50, 50), axis=ax, products=products, number=1, product=product, axis_p=ax[0][0],
        #            tau=tau, p=p, S=S)
        #
        # plot_t_cut(indexes=(50, 100), axis=ax, products=products, number=2, product=product, axis_p=ax[0][0],
        #            tau=tau, p=p, S=S)
        #
        # plot_t_cut(indexes=(50, 120), axis=ax, products=products, number=3, product=product, axis_p=ax[0][0],
        #            tau=tau, p=p, S=S)

        ax[0][0].set_ylabel('e_vec')
        ax[0][0].set_xlabel('tau')
        ax[0][0].pcolormesh(ir.tmat, e_vec, S, cmap='jet')

    return ir.tmat, e_vec, S






xuv = XUV_Field(N=128)
ir = IR_Field(N=128)





fig, ax = plt.subplots(2, 2, figsize=(6, 6))
plt.subplots_adjust(left=0.07, right=0.97)
ax[0][0].plot(ir.tmat, ir.Et, color='blue')
ax[0][0].set_xlabel('time [Femtoseconds]')
ax[0][0].set_title('IR field')
# ax[0][0].text(0.6, 0.8, 'FWHM:'+ str(ir.fwhm/1e-15) +'[fs]\n$\omega_0$: '+ str(ir.w0/1e15) + r'$\cdot 10^{15} \frac{rad}{s}$',
#               transform=ax[0][0].transAxes, backgroundcolor='white')

ax[0][1].plot(xuv.tmat, xuv.Et, color='orange')
ax[0][1].set_xlabel('time [Attoseconds]')
ax[0][1].set_title('Attosecond Pulse')
# ax[0][1].text(0.6, 0.8, 'FWHM:'+ str(xuv.fwhm/1e-18) +'[As]\n$\omega_0$: '+ str(xuv.w0/1e18) + r'$\cdot 10^{18} \frac{rad}{s}$',
#               transform=ax[0][1].transAxes, backgroundcolor='white')

ax[1][0].plot(ir.tmat, ir.Et, color='blue')
ax[1][0].plot(xuv.tmat, xuv.Et, color='orange')
ax[1][0].set_xlabel('time [Femtoseconds]')

tau, p, S = attosecond_streak(xuv=xuv, ir=ir, plot=True)
ax[1][1].pcolormesh(tau, p, S, cmap='jet')
ax[1][1].set_xlabel('delay [Femtoseconds]')
ax[1][1].set_ylabel('momentum')


plt.show()