import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.constants as sc
import time


# SI units for defining parameters
W = 1
cm = 1e-2
um = 1e-6
fs = 1e-15
atts = 1e-18

class XUV_Field():

    def __init__(self, N, tmax):

        # define parameters in SI units
        self.N = N
        # self.en0 = 20 * sc.eV # central energy
        # self.den0 = 75 * sc.eV #energy fwhm
        self.f0 = 80e15
        self.T0 = 1/self.f0 # optical cycle
        self.t0 = 20e-18 # pulse duration
        self.gdd = 0 * atts**2 # gdd
        self.gdd_si = self.gdd / atts**2
        self.tod = -2000 * atts**3 # TOD
        self.tod_si = self.tod / atts**3

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
        self.t0 = self.t0 / sc.physical_constants['atomic unit of time'][0]
        self.f0 = self.f0 * sc.physical_constants['atomic unit of time'][0]
        self.T0 = self.T0 / sc.physical_constants['atomic unit of time'][0]
        self.gdd = self.gdd / sc.physical_constants['atomic unit of time'][0]**2
        self.tod = self.tod / sc.physical_constants['atomic unit of time'][0]**3
        self.dt = self.dt / sc.physical_constants['atomic unit of time'][0]
        self.tmat = self.tmat / sc.physical_constants['atomic unit of time'][0]
        self.fmat = self.fmat * sc.physical_constants['atomic unit of time'][0]
        self.enmat = self.enmat / sc.physical_constants['atomic unit of energy'][0]

        # set up streaking xuv field in AU
        self.Et = np.exp(-2 * np.log(2) * (self.tmat/self.t0)**2 ) * np.exp(2j * np.pi * self.f0 * self.tmat)

        # add GDD to streaking XUV field
        Ef = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.Et)))
        Ef_prop = Ef * np.exp(1j * 0.5 * self.gdd * (2 * np.pi)**2 * (self.fmat - self.f0)**2)
        # plt.figure(98)
        # plt.plot(0.5 * self.gdd * (2 * np.pi)**2 * (self.fmat - self.f0)**2)

        # add TOD to streaking XUV field
        # plt.figure(99)
        # plt.plot(0.5 * self.tod * (2 * np.pi)**3 * (self.fmat - self.f0)**3)
        # plt.figure(100)
        # plt.plot(np.real(Ef_prop), color='blue')
        # plt.plot(np.imag(Ef_prop), color='red')
        Ef_prop = Ef_prop * np.exp(1j * 0.5 * self.tod * (2 * np.pi)**3 * (self.fmat - self.f0)**3)
        # plt.figure(101)
        # plt.plot(np.real(Ef_prop), color='blue')
        # plt.plot(np.imag(Ef_prop), color='red')

        self.Et = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ef_prop)))


class IR_Field():

    def __init__(self, N, tmax):
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

        self.Ip = 24.587 * sc.electron_volt
        self.Ip = self.Ip / sc.physical_constants['atomic unit of energy'][0]


N = 2**16
tmax = 60e-15
xuv = XUV_Field(N=N, tmax=tmax)
ir = IR_Field(N=N, tmax=tmax)
med = Med()

# construct delay axis
N = ir.N
df = 1 / (ir.dt * N)

dt = ir.dt
fvec = df * np.arange(-N/2, N/2, 1)


# construct frequency vector for plotting
f_scale = 2
frequency_positive = f_scale * fvec[int(len(fvec)/2):]

# construct the delay vector and momentum vector for plotting
span = 300
trim = 20
skip = 120
p_vec = np.sqrt(2 * frequency_positive)
tvec =  ir.tmat
tauvec = ir.tmat_indexes[int(trim*span):-int(trim*span)]
tauvec = tauvec[::skip]
p_vec = p_vec[::skip]

# calculate At
A_t = -1 * dt * np.cumsum(ir.Et)
A_t_integ = -1 * np.flip(dt * np.cumsum(np.flip(A_t, axis=0)), axis=0)

# vectors used for calculation
items = {'A_t_integ': A_t_integ, 'Exuv': xuv.Et, 'Ip': med.Ip, 't': tvec}



middle_index = int(len(xuv.Et) / 2)
lower = middle_index-int(span/2)
upper = middle_index+int(span/2)
indexes_zero_delay = lower + np.array(range(upper-lower))
indexes = indexes_zero_delay.reshape(-1, 1) + tauvec.reshape(1, -1)
A_integrals = np.array([np.take(items['A_t_integ'], indexes)])
t_vals = np.array([np.take(items['t'], indexes)])

plt.figure(4)
plt.plot(np.real(xuv.Et[lower:upper]))
plt.show()


p = p_vec.reshape(-1, 1, 1)
p_A_int_mat = np.exp(1j * p * A_integrals)


K = (0.5 * p**2)
e_fft = np.exp(-1j * (K + items['Ip']) * t_vals)

# convert values to tensorflow
xuv_input = tf.placeholder(tf.complex64, [1, span, 1])

p_A_int_mat_tf = tf.constant(p_A_int_mat, dtype=tf.complex64)

e_fft_tf = tf.constant(e_fft, dtype=tf.complex64)

del p_A_int_mat
del e_fft

product = xuv_input * p_A_int_mat_tf * e_fft_tf

integral = tf.constant(dt, dtype=tf.complex64) * tf.reduce_sum(product, axis=1)

image = tf.square(tf.abs(integral))

init = tf.global_variables_initializer()

with tf.Session() as sess:

    init.run()

    strace = sess.run(image, feed_dict={xuv_input: xuv.Et[lower:upper].reshape(1, -1, 1)})

    plt.figure(1)
    plt.pcolormesh(strace, cmap='jet')
    plt.show()






































