import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.constants as sc
import time
import pickle


# SI units for defining parameters
W = 1
cm = 1e-2
um = 1e-6
fs = 1e-15
atts = 1e-18

class XUV_Field():

    def __init__(self, N, tmax, gdd, tod):

        # smaller dt

        # define parameters in SI units
        self.N = N
        # self.en0 = 20 * sc.eV # central energy
        # self.den0 = 75 * sc.eV #energy fwhm
        self.f0 = 120e15
        self.T0 = 1/self.f0 # optical cycle
        # self.t0 = 5e-18 # pulse duration
        self.bandwidth = 350 * sc.electron_volt # bandwidth, joules
        self.gdd = gdd * atts**2 # gdd
        self.gdd_si = self.gdd / atts**2
        self.tod = tod * atts**3 # TOD
        self.tod_si = self.tod / atts**3

        # number of central time steps to integrate
        self.span = 512

        #discretize
        self.tmax = tmax
        self.dt = self.tmax / N
        self.tmat = self.dt * np.arange(-N/2, N/2, 1)

        # discretize the streaking xuv field spectral matrix
        self.df = 1/(self.dt * N)
        self.fmat = self.df * np.arange(-N/2, N/2, 1)
        self.enmat = sc.h * self.fmat

        # convert to AU
        # self.en0 = self.en0 / sc.physical_constants['atomic unit of energy'][0]
        # self.den0 = self.den0 / sc.physical_constants['atomic unit of energy'][0]

        #convert bandwidth from joules to Hz
        self.bandwidth = self.bandwidth /sc.h #Hz
        # convert bandwidth to atomic units
        self.bandwidth = self.bandwidth * sc.physical_constants['atomic unit of time'][0]


        # self.t0 = self.t0 / sc.physical_constants['atomic unit of time'][0]
        self.f0 = self.f0 * sc.physical_constants['atomic unit of time'][0]
        self.T0 = self.T0 / sc.physical_constants['atomic unit of time'][0]
        self.gdd = self.gdd / sc.physical_constants['atomic unit of time'][0]**2
        self.tod = self.tod / sc.physical_constants['atomic unit of time'][0]**3
        self.dt = self.dt / sc.physical_constants['atomic unit of time'][0]
        self.tmat = self.tmat / sc.physical_constants['atomic unit of time'][0]
        self.fmat = self.fmat * sc.physical_constants['atomic unit of time'][0]
        self.enmat = self.enmat / sc.physical_constants['atomic unit of energy'][0]

        # set up streaking xuv field in AU
        # self.Et = np.exp(-2 * np.log(2) * (self.tmat/self.t0)**2 ) * np.exp(2j * np.pi * self.f0 * self.tmat)

        # add GDD/TOD to streaking XUV field
        # Ef = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.Et)))

        Ef0 = np.exp(-2 * np.log(2) * ((self.fmat - self.f0) / self.bandwidth)**2)
        Ef1 = 0.6 * np.exp(-2 * np.log(2) * ((self.fmat - (self.f0 - 1.3)) / (self.bandwidth/2))**2)
        Ef2 = 0.6 * np.exp(-2 * np.log(2) * ((self.fmat - (self.f0 + 1.3)) / (self.bandwidth / 2)) ** 2)
        Ef = Ef0 + Ef1 + Ef2
        # Ef = Ef0
        self.Ef_prop = Ef * np.exp(1j * 0.5 * self.gdd * (2 * np.pi)**2 * (self.fmat - self.f0)**2)
        self.Ef_prop = self.Ef_prop * np.exp(1j * 0.5 * self.tod * (2 * np.pi)**3 * (self.fmat - self.f0)**3)

        self.Et = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(self.Ef_prop)))


class IR_Field():

    def __init__(self, N, tmax):
        self.N = N
        # calculate parameters in SI units
        self.lam0 = 1.7 * um    # central wavelength
        self.f0 = sc.c/self.lam0    # carrier frequency
        self.T0 = 1/self.f0 # optical cycle
        self.t0 = 12 * fs # pulse duration
        self.ncyc = self.t0/self.T0
        self.I0_si = 1e13 # W/cm**2
        self.I0 = self.I0_si * W/cm**2

        # compute ponderomotive energy
        self.Up = (sc.elementary_charge**2 * self.I0) / (2 * sc.c * sc.epsilon_0 * sc.electron_mass * (2 * np.pi * self.f0)**2)

        # discretize time matrix
        self.tmax = tmax
        self.dt = self.tmax / N
        self.tmat = self.dt * np.arange(-N/2, N/2, 1)
        self.tmat_indexes = np.arange(int(-N/2), int(N/2), 1)

        # discretize spectral matrix
        self.df = 1/(self.dt * N)
        self.fmat = self.df * np.arange(-N/2, N/2, 1)
        self.enmat = sc.h * self.fmat

        # convert units to AU
        self.t0 = self.t0 / sc.physical_constants['atomic unit of time'][0]
        self.f0 = self.f0 * sc.physical_constants['atomic unit of time'][0]
        self.df = self.df * sc.physical_constants['atomic unit of time'][0]

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

        # self.Ip_eV = 24.587
        self.Ip_eV = 13.6
        self.Ip = self.Ip_eV * sc.electron_volt # joules
        self.Ip = self.Ip / sc.physical_constants['atomic unit of energy'][0] # a.u.


def plot_spectrum(xuv_Ef, xuv_fmat):
    # plot also the spectrum
    xuv_fmat_HZ = xuv_fmat / sc.physical_constants['atomic unit of time'][0]

    # frequency in Hz
    joules = sc.h * xuv_fmat_HZ # joules
    electronvolts = 1 / (sc.elementary_charge) * joules

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)

    # plot the modulus of spectrum in eV
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(electronvolts, np.abs(xuv_Ef), color='black')
    ax.set_title('|E|')
    ax.set_xlabel('eV')
    ax.set_xlim(100, 600)


    ax = fig.add_subplot(gs[1, 0])
    ax.plot(electronvolts, np.abs(xuv_Ef)**2, color='black')
    ax.set_title('I')
    ax.set_xlabel('eV')
    ax.set_xlim(100, 600)


    # plot the modulus of spectrum in Hz
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(xuv_fmat_HZ, np.abs(xuv_Ef), color='black')
    ax.set_title('|E|')
    ax.set_xlabel('HZ')
    ax.set_xlim(0, 2e17)

    # plot the modulus of spectrum in Hz
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(xuv_fmat_HZ, np.abs(xuv_Ef)**2, color='black')
    ax.set_title('I')
    ax.set_xlabel('HZ')
    ax.set_xlim(0, 2e17)

    # plot just the spectrum intensity

    # find FWHM
    I = np.abs(xuv_Ef)**2
    maximum_energy = np.max(I)
    fwhm_i1 = np.argmin(np.abs(maximum_energy/2 - I))
    I_2 = np.array(I)
    I_2[fwhm_i1] = 0
    fwhm_i2 = np.argmin(np.abs(maximum_energy / 2 - I_2))


    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[:, :])

    # plot the dots
    ax.plot(electronvolts[fwhm_i1], I[fwhm_i1], 'r*')
    ax.plot(electronvolts[fwhm_i2], I[fwhm_i2], 'r*')

    # plot the line
    ax.plot([electronvolts[fwhm_i1], electronvolts[fwhm_i2]], [I[fwhm_i1], I[fwhm_i2]], 'r')

    # calculate FWHM
    fwhm = np.abs(electronvolts[fwhm_i1] - electronvolts[fwhm_i2])

    ax.plot(electronvolts, I, color='black')
    ax.text(50, I[fwhm_i1], 'FWHM: '+str(round(fwhm, 1))+' eV', backgroundcolor='white')
    ax.set_title('Intensity')
    ax.set_xlabel('eV')
    ax.set_ylabel('Intensity')
    ax.set_xlim(0, 1200)


def plot_xuv_ir_atto_1(save):

    fig, ax = plt.subplots(2, 2, figsize=(10,8))

    si_time = tauvec * dt * sc.physical_constants['atomic unit of time'][0]
    # ax[1][1].pcolormesh(tauvec*dt, p_vec, strace, cmap='jet')

    # convert p_vec to eV
    energy_values = sc.physical_constants['atomic unit of energy'][0] * 0.5 * p_vec**2 # joules
    energy_values = energy_values / sc.electron_volt # electron volts

    ax[1][1].pcolormesh(si_time*1e15, energy_values, strace, cmap='jet')
    ax[1][1].text(0.05, 0.95, '$I_p$: {} eV'.format(med.Ip_eV), backgroundcolor='white', transform=ax[1][1].transAxes)
    ax[1][1].set_xlabel('fs')
    ax[1][1].set_ylabel('eV')

    si_time = xuv_int_t * sc.physical_constants['atomic unit of time'][0]
    # ax[0][0].plot(xuv_int_t, np.real(xuv_integral_space), color='blue')
    ax[0][0].plot(si_time*1e18, np.real(xuv_integral_space), color='blue')
    ax[0][0].set_xlabel('as')

    si_time = ir.tmat * sc.physical_constants['atomic unit of time'][0]
    # ax[0][1].plot(ir.tmat, ir.Et, color='orange')
    ax[0][1].plot(si_time*1e15, ir.Et, color='orange')
    ax[0][1].set_xlabel('fs')

    si_time = ir.tmat * sc.physical_constants['atomic unit of time'][0]
    # ax[1][0].plot(ir.tmat, ir.Et, color='orange')
    ax[1][0].plot(si_time*1e15, ir.Et, color='orange')
    ax[1][0].set_xlabel('fs')

    axtwin = ax[1][0].twinx()
    si_time = xuv_int_t * sc.physical_constants['atomic unit of time'][0]
    # axtwin.plot(xuv_int_t, np.real(xuv_integral_space), color='blue')
    axtwin.plot(si_time*1e15, np.real(xuv_integral_space), color='blue')
    axtwin.set_xlabel('fs')
    if save:
        plt.savefig('./xuv_ir_atto_Ip{}.png'.format(str(med.Ip_eV)))

    if save:
        plt.figure(10)
        plt.pcolormesh(tauvec*dt, p_vec, strace, cmap='jet')
        plt.text(0.1, 0.92, 'GDD: {}'.format(xuv.gdd_si) + ' $as^2$',
                 transform=plt.gca().transAxes, backgroundcolor='white')
        plt.text(0.1, 0.85, 'TOD: {}'.format(xuv.tod_si) + ' $as^3$',
                 transform=plt.gca().transAxes, backgroundcolor='white')
        plt.savefig('./tracegdd{}tod{}.png'.format(int(xuv.gdd_si), int(xuv.tod_si)))



def complete_plot(xuv, xuv_fmat, ir_intensity):

    xuv_Ef = xuv.Ef_prop
    fig = plt.figure(figsize=(15,10))
    gs = fig.add_gridspec(2,3)


    # plot the attosecond pulse
    ax = fig.add_subplot(gs[0,0])
    si_time = xuv_int_t * sc.physical_constants['atomic unit of time'][0]
    ax.plot(si_time * 1e18, np.real(xuv_integral_space), color='blue', label='Real $E_{xuv}$')
    ax.text(0.1, 0.9, 'XUV pulse', transform=ax.transAxes, backgroundcolor='white')
    ax.legend()
    ax.set_xlabel('attoseconds')
    #label GDD and TOD
    ax.text(0.1,0.8,'GDD: '+str(xuv.gdd_si)+' $as^2$', transform=ax.transAxes)
    ax.text(0.1,0.7,'TOD: '+str(xuv.tod_si)+' $as^3$', transform=ax.transAxes)


    # plot the attosecond pulse intensity
    ax = fig.add_subplot(gs[0, 1])
    si_time = xuv_int_t * sc.physical_constants['atomic unit of time'][0]
    plot_time = si_time * 1e18
    Ixuv_t = np.abs(xuv_integral_space)**2
    max_I_t = np.max(Ixuv_t)
    fwhm_i1 = np.argmin(np.abs(max_I_t/2 - Ixuv_t))
    I_2 = np.array(Ixuv_t)
    I_2[fwhm_i1] = 0
    fwhm_i2 = np.argmin(np.abs(max_I_t / 2 - I_2))
    # plot the points
    ax.plot(plot_time[fwhm_i1], Ixuv_t[fwhm_i1], 'r*')
    ax.plot(plot_time[fwhm_i2], Ixuv_t[fwhm_i2], 'r*')
    # plot the line
    ax.plot([plot_time[fwhm_i1], plot_time[fwhm_i2]], [Ixuv_t[fwhm_i1], Ixuv_t[fwhm_i2]], 'r')
    # calculate FWHM
    fwhm = np.abs(plot_time[fwhm_i1] - plot_time[fwhm_i2])
    ax.text(0.1, 0.1, 'FWHM: ' + str(round(fwhm, 1)) + ' as', backgroundcolor='white', transform=ax.transAxes, color='red')
    # plot the Intensity
    ax.plot(plot_time, Ixuv_t, color='black', label='$I_{xuv}$')
    ax.text(0.1, 0.9, 'XUV pulse\nIntensity', transform=ax.transAxes, backgroundcolor='white')
    # ax.legend()
    ax.set_yticks([])
    ax.set_xlabel('attoseconds')


    # plot the IR
    ax = fig.add_subplot(gs[1,0])
    si_time = ir.tmat * sc.physical_constants['atomic unit of time'][0]
    ax.plot(si_time * 1e15, ir.Et, color='orange', label='$E_{IR}$')
    ax.text(0.1, 0.9, 'IR pulse', transform=ax.transAxes, backgroundcolor='white')
    ax.text(0.1, 0.1, '$I_{IR}$:'+str(int(ir_intensity/1e12))+r'$\cdot 10^{12} \frac{W}{cm^{2}}$',
            backgroundcolor='white', transform=ax.transAxes)
    ax.set_ylabel('IR [atomic units]')
    ax.set_xlabel('fs')


    # plot the streaking trace
    ax = fig.add_subplot(gs[1, 1:3])
    si_time = tauvec * dt * sc.physical_constants['atomic unit of time'][0]
    # convert p_vec to eV
    energy_values = sc.physical_constants['atomic unit of energy'][0] * 0.5 * p_vec ** 2  # joules
    energy_values = energy_values / sc.electron_volt  # electron volts
    ax.pcolormesh(si_time * 1e15, energy_values, strace, cmap='jet')
    ax.text(0.1, 0.9, '$I_p$: {} eV'.format(med.Ip_eV), backgroundcolor='white', transform=ax.transAxes)
    ax.set_xlabel('fs')
    ax.set_ylabel('eV')


    # plot the XUV spectrum
    ax = fig.add_subplot(gs[0, 2])
    # plot also the spectrum
    xuv_fmat_HZ = xuv_fmat / sc.physical_constants['atomic unit of time'][0]
    # frequency in Hz
    joules = sc.h * xuv_fmat_HZ  # joules
    electronvolts = 1 / (sc.elementary_charge) * joules
    # find FWHM
    I = np.abs(xuv_Ef) ** 2
    maximum_energy = np.max(I)
    fwhm_i1 = np.argmin(np.abs(maximum_energy / 2 - I))
    I_2 = np.array(I)
    I_2[fwhm_i1] = 0
    fwhm_i2 = np.argmin(np.abs(maximum_energy / 2 - I_2))
    ax.plot(electronvolts[fwhm_i1], I[fwhm_i1], 'r*')
    ax.plot(electronvolts[fwhm_i2], I[fwhm_i2], 'r*')
    # plot the line
    ax.plot([electronvolts[fwhm_i1], electronvolts[fwhm_i2]], [I[fwhm_i1], I[fwhm_i2]], 'r')
    # calculate FWHM
    fwhm = np.abs(electronvolts[fwhm_i1] - electronvolts[fwhm_i2])
    #find 0 index
    zero_eV_index = np.argmin(np.abs(electronvolts - 0))
    min_index, max_index = zero_eV_index, -50000
    ax.plot(electronvolts[min_index:max_index], I[min_index:max_index], color='black', label='$I_{xuv}$')
    ax.text(0.1, 0.9, 'XUV spectrum', transform=ax.transAxes, backgroundcolor='white')
    ax.set_xlabel('eV')
    ax.set_ylabel('Intensity')
    # ax.set_xlim(0, 1200)
    # plot the phase
    axtwin = ax.twinx()
    axtwin.plot(electronvolts[min_index:max_index], np.unwrap(np.angle(xuv_Ef[min_index:max_index])), color='green', alpha=0.5, linestyle='dashed')
    space = 0.5
    axtwin.set_ylim(np.min(np.unwrap(np.angle(xuv_Ef[min_index:max_index])))-0.5*space, np.max(np.unwrap(np.angle(xuv_Ef[min_index:max_index])))+space)
    axtwin.tick_params(colors='green')
    axtwin.set_ylabel('$\phi (eV)$', color='green')
    axtwin.text(0.1, 0.1, 'FWHM: ' + str(round(fwhm, 1)) + ' eV', backgroundcolor='white', transform=ax.transAxes,
            color='red')
    plt.savefig('./{}gdd.png'.format(str(xuv.gdd_si)))



N = 2**17
tmax = 60e-15
xuv = XUV_Field(N=N, tmax=tmax, gdd=0, tod=0)
ir = IR_Field(N=N, tmax=tmax)
med = Med()

# construct delay axis
N = ir.N
df = 1 / (ir.dt * N)

dt = ir.dt
fvec = df * np.arange(-N/2, N/2, 1)


# construct the delay vector and momentum vector for plotting
span = xuv.span
tvec =  ir.tmat

# p_vec = np.linspace(3, 6.5, 200) #a.u.
# tauvec = np.arange(-22000, 22000, 250) #indexes

p_max, p_min = 9, 0
k_max, k_min = 0.5*p_max**2, 0.5*p_min**2
k_vec = np.linspace(k_min, k_max, 200)
# k_vec = np.linspace(k_min, k_max, 30)
p_vec = np.sqrt(2 * k_vec)


tau_vals = [-22000, 22000, 250]
# factor by two to make the time steps of the delay correct for the IR timescale
tau_vals  = [2 * val for val in tau_vals]
tauvec = np.arange(tau_vals[0], tau_vals[1], tau_vals[2]) #indexes
# tauvec = np.arange(-22000, 22000, 250) #indexes

# few cycle
# p_vec = np.linspace(3, 6.5, 250)
# tauvec = np.arange(-5000, 5000, 56)

# calculate At
A_t = -1 * dt * np.cumsum(ir.Et)
A_t_integ = -1 * np.flip(dt * np.cumsum(np.flip(A_t, axis=0)), axis=0)

# vectors used for calculation
items = {'A_t_integ': A_t_integ, 'Exuv': xuv.Et, 'Ip': med.Ip, 't': tvec}

# find the time/vectors at the various delay steps
middle_index = int(len(xuv.Et) / 2)
lower = middle_index-int(span/2)
upper = middle_index+int(span/2)
indexes_zero_delay = lower + np.array(range(upper-lower))
indexes = indexes_zero_delay.reshape(-1, 1) + tauvec.reshape(1, -1)
A_integrals = np.array([np.take(items['A_t_integ'], indexes)])
t_vals = np.array([np.take(items['t'], indexes)])

xuv_integral_space = xuv.Et[lower:upper]
xuv_int_t = xuv.tmat[lower:upper]


p = p_vec.reshape(-1, 1, 1)
p_A_int_mat = np.exp(1j * p * A_integrals)


K = (0.5 * p**2) # a.u.
e_fft = np.exp(-1j * (K + items['Ip']) * t_vals)

# convert values to tensorflow
xuv_input = tf.placeholder(tf.complex64, [1, span, 1])
p_A_int_mat_tf = tf.constant(p_A_int_mat, dtype=tf.complex64)
e_fft_tf = tf.constant(e_fft, dtype=tf.complex64)

# free memory
del p_A_int_mat
del e_fft

product = xuv_input * p_A_int_mat_tf * e_fft_tf

integral = tf.constant(dt, dtype=tf.complex64) * tf.reduce_sum(product, axis=1)

image = tf.square(tf.abs(integral))

# write items needed to pickle becasue im tired of loading this whole thing
with open('crab_tf_items.p', 'wb') as file:

    crab_tf_items = {}
    crab_tf_items['items'] = items
    crab_tf_items['xuv_int_t'] = xuv_int_t
    crab_tf_items['N'] = N
    crab_tf_items['tmax'] = tmax
    crab_tf_items['dt'] = dt
    crab_tf_items['tauvec'] = tauvec
    crab_tf_items['p_vec'] = p_vec
    crab_tf_items['irf0'] = ir.f0
    crab_tf_items['irEt'] = ir.Et
    crab_tf_items['irtmat'] = ir.tmat
    crab_tf_items['xuvf0'] = xuv.f0


    pickle.dump(crab_tf_items, file)
    print('files pickled')




if __name__ == '__main__':

    plot = True
    save = True

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        init.run()

        time1 = time.time()
        strace = sess.run(image, feed_dict={xuv_input: xuv_integral_space.reshape(1, -1, 1)})
        time2 = time.time()
        duration = time2 - time1
        print("duration: ", duration)

        if plot:

            # plot_spectrum(xuv_Ef=xuv.Ef_prop, xuv_fmat=xuv.fmat)
            # plot_xuv_ir_atto_1(save)
            complete_plot(xuv=xuv, xuv_fmat=xuv.fmat, ir_intensity=ir.I0_si)

            plt.show()


