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

    def __init__(self, N, tmax, start_index, end_index, gdd=0.0, tod=0.0, random_phase=None):

        # define parameters in SI units
        self.N = N
        self.f0 = 80e15
        self.T0 = 1/self.f0 # optical cycle
        self.t0 = 20e-18 # pulse duration
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
        self.t0 = self.t0 / sc.physical_constants['atomic unit of time'][0]
        self.f0 = self.f0 * sc.physical_constants['atomic unit of time'][0]
        self.T0 = self.T0 / sc.physical_constants['atomic unit of time'][0]
        self.gdd = self.gdd / sc.physical_constants['atomic unit of time'][0]**2
        self.tod = self.tod / sc.physical_constants['atomic unit of time'][0]**3
        self.dt = self.dt / sc.physical_constants['atomic unit of time'][0]
        self.tmat = self.tmat / sc.physical_constants['atomic unit of time'][0]
        self.fmat = self.fmat * sc.physical_constants['atomic unit of time'][0]
        self.enmat = self.enmat / sc.physical_constants['atomic unit of energy'][0]

        # calculate bandwidth from fwhm
        self.bandwidth = 0.44 / self.t0

        Ef = np.exp(-2 * np.log(2) * ((self.fmat - self.f0) / self.bandwidth) ** 2)

        # apply the TOD and GDD phase if specified
        phi = (1/2) * self.gdd * (2 * np.pi)**2 * (self.fmat - self.f0)**2 + (1/6) * self.tod * (2 * np.pi)**3 * (self.fmat - self.f0)**3
        self.Ef_prop = Ef * np.exp(1j * phi)

        # apply the random phase if specified
        if random_phase:
            print('apply random phase')

        self.Et_prop = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(self.Ef_prop)))

        self.Ef_prop_cropped = self.Ef_prop[start_index:end_index]
        self.f_cropped = self.fmat[start_index:end_index]


class IR_Field():

    def __init__(self, N, tmax, start_index, end_index):
        self.N = N
        # calculate parameters in SI units
        self.lam0 = 1.7 * um    # central wavelength
        self.f0 = sc.c/self.lam0    # carrier frequency
        self.T0 = 1/self.f0 # optical cycle
        # self.t0 = 12 * fs # pulse duration
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
        self.Et = self.E0 * np.exp(-2 * np.log(2) * (self.tmat/self.t0)**2) * np.exp(1j * 2 * np.pi * self.f0 * self.tmat)

        # fourier transform the field
        self.Ef = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.Et)))

        # add phase ... later
        self.Ef_prop = self.Ef

        # fourier transform back to time domain
        self.Et_prop = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(self.Ef_prop)))


        # crop the field for input
        self.Ef_prop_cropped = self.Ef[start_index:end_index]
        self.f_cropped = self.fmat[start_index:end_index]


class Med():

    def __init__(self):
        self.Ip_eV = 24.587
        self.Ip = self.Ip_eV * sc.electron_volt  # joules
        self.Ip = self.Ip / sc.physical_constants['atomic unit of energy'][0]  # a.u.



def plot_initial_field(field, timespan):
    fig = plt.figure()
    gs = fig.add_gridspec(3, 2)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(field.tmat, np.real(field.Et_prop), color='blue')
    ax.plot(field.tmat, np.imag(field.Et_prop), color='red')
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(field.fmat, np.real(field.Ef_prop), color='blue')
    ax.plot(field.fmat, np.imag(field.Ef_prop), color='red')
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(field.f_cropped, np.real(field.Ef_prop_cropped), color='blue')
    ax.plot(field.f_cropped, np.imag(field.Ef_prop_cropped), color='red')
    ax.text(0, -0.25, 'cropped frequency ({} long)'.format(int(timespan)), transform=ax.transAxes,
            backgroundcolor='white')


# use these indexes to crop the ir and xuv frequency space for input to the neural net
xuv_fmin_index,  xuv_fmax_index = 270, 325
ir_fmin_index, ir_fmax_index = 64, 84

# the length of each vector, ir and xuv
xuv_frequency_grid_length = xuv_fmax_index - xuv_fmin_index
ir_frequency_grid_length = ir_fmax_index - ir_fmin_index


# create two time axes, with the same dt for the xuv and the IR
xuv = XUV_Field(N=512, tmax=5e-16, start_index=xuv_fmin_index, end_index=xuv_fmax_index)
ir = IR_Field(N=128, tmax=50e-15, start_index=ir_fmin_index, end_index=ir_fmax_index)

# plot the xuv field
plot_initial_field(field=xuv, timespan=int(xuv_frequency_grid_length))

# plot the infrared field
plot_initial_field(field=ir, timespan=int(ir_frequency_grid_length))

# construct the field with tensorflow

# placeholders
xuv_cropped_f = tf.placeholder(tf.complex64, [1, len(xuv.Ef_prop_cropped)])
ir_cropped_f = tf.placeholder(tf.complex64, [len(ir.Ef_prop_cropped)])

# constants
xuv_fmat = tf.constant(xuv.fmat, dtype=tf.float32)
ir_fmat = tf.constant(ir.fmat, dtype=tf.float32)


# zero pad the spectrum of ir and xuv input
paddings = tf.constant([[0,0], [1,2]], dtype=tf.int32)

padded = tf.pad(xuv_cropped_f, paddings)


init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    out = sess.run(padded, feed_dict={xuv_cropped_f: xuv.Ef_prop_cropped.reshape(1,-1)})
    print(out)
    exit(0)













plt.show()
