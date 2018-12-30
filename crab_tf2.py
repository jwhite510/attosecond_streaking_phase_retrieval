import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.constants as sc
import time
import pickle
import math
from scipy.interpolate import interp1d
import math


# SI units for defining parameters
W = 1
cm = 1e-2
um = 1e-6
fs = 1e-15
atts = 1e-18



class XUV_Field():

    def __init__(self, N=512, tmax=5e-16, start_index=270, end_index=325, gdd=0.0, tod=0.0,
                 random_phase=None, measured_spectrum=None, random_phase_taylor=None,
                 f0=80e15, t0=20e-18):

        if not measured_spectrum:

            # start and end indexes
            self.fmin_index = start_index
            self.fmax_index = end_index

            # define parameters in SI units
            self.N = N
            self.f0 = f0
            self.T0 = 1/self.f0 # optical cycle
            self.t0 = t0 # pulse duration
            self.gdd = gdd * atts**2 # gdd
            self.gdd_si = self.gdd / atts**2
            self.tod = tod * atts**3 # TOD
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
            self.t0 = self.t0 / sc.physical_constants['atomic unit of time'][0]
            self.f0 = self.f0 * sc.physical_constants['atomic unit of time'][0]
            self.T0 = self.T0 / sc.physical_constants['atomic unit of time'][0]
            self.gdd = self.gdd / sc.physical_constants['atomic unit of time'][0]**2
            self.tod = self.tod / sc.physical_constants['atomic unit of time'][0]**3
            self.dt = self.dt / sc.physical_constants['atomic unit of time'][0]
            self.df = self.df * sc.physical_constants['atomic unit of time'][0]
            self.tmat = self.tmat / sc.physical_constants['atomic unit of time'][0]
            self.fmat = self.fmat * sc.physical_constants['atomic unit of time'][0]
            self.enmat = self.enmat / sc.physical_constants['atomic unit of energy'][0]

            # calculate bandwidth from fwhm
            self.bandwidth = 0.44 / self.t0

            Ef = np.exp(-2 * np.log(2) * ((self.fmat - self.f0) / self.bandwidth) ** 2)

        else:

            # start and end indexes
            self.fmin_index = measured_spectrum['indexmin']
            self.fmax_index = measured_spectrum['indexmax']

            # define parameters in SI units
            self.N = measured_spectrum['N']
            self.gdd = gdd * atts ** 2  # gdd
            self.gdd_si = self.gdd / atts ** 2
            self.tod = tod * atts ** 3  # TOD
            self.tod_si = self.tod / atts ** 3
            self.f0 = measured_spectrum['f0']

            # discretize
            self.dt = measured_spectrum['dt']
            self.tmat = measured_spectrum['tmat']
            self.fmat = measured_spectrum['fmat']
            self.enmat = sc.h * self.fmat

            # convert to AU
            self.gdd = self.gdd / sc.physical_constants['atomic unit of time'][0] ** 2
            self.f0 = self.f0 * sc.physical_constants['atomic unit of time'][0]
            self.tod = self.tod / sc.physical_constants['atomic unit of time'][0] ** 3
            self.dt = self.dt / sc.physical_constants['atomic unit of time'][0]
            self.tmat = self.tmat / sc.physical_constants['atomic unit of time'][0]
            self.fmat = self.fmat * sc.physical_constants['atomic unit of time'][0]
            self.enmat = self.enmat / sc.physical_constants['atomic unit of energy'][0]

            Ef = measured_spectrum['Ef']


        # apply the TOD and GDD phase if specified
        phi = (1/2) * self.gdd * (2 * np.pi)**2 * (self.fmat - self.f0)**2 + (1/6) * self.tod * (2 * np.pi)**3 * (self.fmat - self.f0)**3
        self.Ef_prop = Ef * np.exp(1j * phi)

        # apply the random phase if specified
        if random_phase:
            # define phase vector
            self.nodes = random_phase['amplitude'] * (np.random.rand(random_phase['nodes']) - 0.5)
            axis_nodes = np.linspace(0, self.N, random_phase['nodes'])
            axis_phase = np.array(range(self.N))
            f = interp1d(axis_nodes, self.nodes, kind='cubic')
            phi = f(axis_phase)
            self.Ef_prop = Ef * np.exp(1j * phi)

        elif random_phase_taylor:
            # generate random phase for
            taylor_coefficients = random_phase_taylor['coefs']
            # generate value for coefficients
            coef_values = np.random.rand(taylor_coefficients) - 0.5
            coef_values = coef_values * random_phase_taylor['amplitude']
            # linear phase always 0
            coef_values[0] = 0
            # gdd set to 0
#            coef_values[1] = 0
            coef_values = coef_values.reshape(-1, 1)

            # calculate factorials
            orders = np.array(range(taylor_coefficients))+1
            terms = np.array(orders).reshape(-1, 1)
            add_thing = np.arange(0, len(terms), 1).reshape(1, -1)
            triangle = np.tril(terms - add_thing)
            triangle[triangle==0] = 1
            exponents = orders.reshape(-1, 1)
            # print(triangle)

            # loop ...
            factorial = np.ones(shape=(taylor_coefficients, 1))
            i = 0
            while i < np.shape(triangle)[1]:
                factorial = factorial * triangle[:, i].reshape(-1, 1)
                i+=1

            # x axis
            taylor_fmat = (self.fmat - self.f0).reshape(1, -1)

            taylor_terms = coef_values * (1/factorial) * taylor_fmat**exponents

            taylor_series = np.sum(taylor_terms, axis=0)

            self.Ef_prop = Ef * np.exp(1j * taylor_series)


        # self.Ef_prop = remove_linear_phase(self.Ef_prop, plotting=False)

        # set phase angle at f0 to 0
        f0_index = np.argmin(np.abs(self.f0-self.fmat))
        f0_angle = np.angle(self.Ef_prop[f0_index])
        self.Ef_prop = self.Ef_prop * np.exp(-1j * f0_angle)


        self.Et_prop = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(self.Ef_prop)))

        self.Ef_prop_cropped = self.Ef_prop[self.fmin_index:self.fmax_index]
        self.f_cropped = self.fmat[self.fmin_index:self.fmax_index]


class IR_Field():

    def __init__(self, N=128, tmax=50e-15, start_index=64, end_index=84, const_phase=0.0, pulse_duration=12.0, clambda=1.7, I0 = 1.0, random_pulse=None):

        # start and end indexes
        self.fmin_index = start_index
        self.fmax_index = end_index

        if random_pulse:
            # define random value between min and max
            # phase:
            self.clambda = random_pulse['clambda_range'][0] + np.random.rand()*(random_pulse['clambda_range'][1] - random_pulse['clambda_range'][0])
            self.pulse_duration = random_pulse['pulse_duration_range'][0] + np.random.rand()*(random_pulse['pulse_duration_range'][1] - random_pulse['pulse_duration_range'][0])
            self.const_phase = random_pulse['phase_range'][0] + np.random.rand()*(random_pulse['phase_range'][1] - random_pulse['phase_range'][0])
            self.I0_sc = random_pulse['I_range'][0] + np.random.rand()*(random_pulse['I_range'][1] - random_pulse['I_range'][0])
            self.I0 = self.I0_sc * 1e13 * W / cm ** 2

        else:

            self.clambda = clambda
            self.pulse_duration = pulse_duration
            self.const_phase = const_phase
            self.I0_sc = I0
            self.I0 = I0 * 1e13 * W / cm ** 2
            #self.I0 = 1 * 1e13 * W / cm ** 2



        self.N = N
        # calculate parameters in SI units
        self.lam0 = self.clambda * um    # central wavelength
        self.f0 = sc.c/self.lam0    # carrier frequency
        self.T0 = 1/self.f0 # optical cycle
        # self.t0 = 12 * fs # pulse duration
        self.t0 = self.pulse_duration * fs # pulse duration
        self.ncyc = self.t0/self.T0



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
        self.Ef_prop = self.Ef * np.exp(1j * self.const_phase)

        # fourier transform back to time domain
        self.Et_prop = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(self.Ef_prop)))


        # crop the field for input
        self.Ef_prop_cropped = self.Ef_prop[self.fmin_index:self.fmax_index]
        self.f_cropped = self.fmat[self.fmin_index:self.fmax_index]


class Med():

    def __init__(self):
        self.Ip_eV = 24.587
        self.Ip = self.Ip_eV * sc.electron_volt  # joules
        self.Ip = self.Ip / sc.physical_constants['atomic unit of energy'][0]  # a.u.


def tf_1d_ifft(tensor, shift, axis=0):

    shifted = tf.manip.roll(tensor, shift=shift, axis=axis)
    # fft
    time_domain_not_shifted = tf.ifft(shifted)
    # shift again
    time_domain = tf.manip.roll(time_domain_not_shifted, shift=shift, axis=axis)

    return time_domain


def tf_1d_fft(tensor, shift, axis=0):

    shifted = tf.manip.roll(tensor, shift=shift, axis=axis)
    # fft
    time_domain_not_shifted = tf.fft(shifted)
    # shift again
    time_domain = tf.manip.roll(time_domain_not_shifted, shift=shift, axis=axis)

    return time_domain


def check_padded_time_domain():
    # check accuracy of padding in time domain

    fig = plt.figure()
    gs = fig.add_gridspec(6, 2)

    ax = fig.add_subplot(gs[0, :])
    # the 256 timestep signal in time
    ax.plot(ir.tmat, np.real(ir.Et_prop))

    # F domain and first padded F domain
    ax = fig.add_subplot(gs[1, :])
    ir_cropped_f_out = sess.run(ir_cropped_f, feed_dict={ir_cropped_f: ir.Ef_prop_cropped})
    ax.plot(ir.f_cropped, np.real(ir_cropped_f_out))
    ax.plot(ir.fmat, np.zeros_like(ir.fmat), alpha=0.1, color='black')

    # the padded F domain 1
    ax = fig.add_subplot(gs[2, :])
    padded_ir_f_out = sess.run(padded_ir_f, feed_dict={ir_cropped_f: ir.Ef_prop_cropped})
    ax.plot(ir.fmat, np.real(padded_ir_f_out))

    # the padded F domain 2
    ax = fig.add_subplot(gs[3, :])
    padded_ir_2_out = sess.run(padded_ir_2, feed_dict={ir_cropped_f: ir.Ef_prop_cropped})
    ax.plot(f_pad_2, np.real(padded_ir_2_out))

    ax = fig.add_subplot(gs[4, :])
    ir_t_matched_dt_out = sess.run(ir_t_matched_dt, feed_dict={ir_cropped_f: ir.Ef_prop_cropped})
    ax.plot(t_pad_2, np.real(ir_t_matched_dt_out))
    # ax.plot(np.real(ir_t_matched_dt_out))

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[:, :])
    # ax.plot(np.real(ir_t_matched_dt_out))
    ax.plot(t_pad_2, np.real(ir_t_matched_dt_out), label='tensorflow')
    ax.legend(loc=3)

    # compare original with smaller dt
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)
    ir_t_matched_dt_scaled_out = sess.run(ir_t_matched_dt_scaled, feed_dict={ir_cropped_f: ir.Ef_prop_cropped})
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t_pad_2, np.real(ir_t_matched_dt_scaled_out), label='matched, scaled')
    # ax.plot(t_pad_2, np.real(ir_t_matched_dt_scaled_out), 'r.')
    ax.plot(ir.tmat, np.real(ir.Et_prop), label='orignal', linestyle='dashed')
    ax.legend(loc=3)


def check_corner_errors():

    fig = plt.figure()
    gs = fig.add_gridspec(6, 2)

    # plot cross section of ir term
    out = sess.run(A_t_integ_t_phase, feed_dict={xuv_cropped_f: xuv.Ef_prop_cropped,
                                                 ir_cropped_f: ir.Ef_prop_cropped})
    ax = fig.add_subplot(gs[0, :])
    ax.pcolormesh(np.real(out[:, :]), cmap='jet')
    ax.text(0, 1, 'A_t_integ_t_phase', transform=ax.transAxes, backgroundcolor='white')

    span = 20
    p_section = 100

    # plot the right side of ir term
    ax = fig.add_subplot(gs[1, 1])
    ax.pcolormesh(np.real(out[:, -span:]), cmap='jet')

    # plot the left side of ir term
    ax = fig.add_subplot(gs[1, 0])
    ax.pcolormesh(np.real(out[:, :span]), cmap='jet')

    # plot the cross section of ir_phi term
    out = sess.run(ir_phi, feed_dict={xuv_cropped_f: xuv.Ef_prop_cropped,
                                      ir_cropped_f: ir.Ef_prop_cropped})
    ax = fig.add_subplot(gs[2, :])
    ax.pcolormesh(np.real(out[p_section, :, :]), cmap='jet')
    ax.text(0, 1, 'ir_phi', transform=ax.transAxes, backgroundcolor='white')

    # plot the left and right side of the ir phi term
    ax = fig.add_subplot(gs[3, 0])
    ax.pcolormesh(np.real(out[p_section, :, :span]), cmap='jet')

    ax = fig.add_subplot(gs[3, 1])
    ax.pcolormesh(np.real(out[p_section, :, -span:]), cmap='jet')

    # plot the cross section of the fourier transform
    out = sess.run(e_fft_tf, feed_dict={xuv_cropped_f: xuv.Ef_prop_cropped,
                                        ir_cropped_f: ir.Ef_prop_cropped})
    ax = fig.add_subplot(gs[4, :])
    ax.plot(np.real(out[p_section, :, 0]))

    # plot the streaking trace
    out = sess.run(image, feed_dict={xuv_cropped_f: xuv.Ef_prop_cropped,
                                     ir_cropped_f: ir.Ef_prop_cropped})
    ax = fig.add_subplot(gs[5, :])
    ax.pcolormesh(out, cmap='jet')
    plt.savefig('./corners_const_phase/128/64bitfloat/6.png')


def check_fft_and_reconstruction():

    out_xuv = sess.run(padded_xuv_f, feed_dict={xuv_cropped_f: xuv.Ef_prop_cropped})
    out_xuv_time = sess.run(xuv_time_domain, feed_dict={xuv_cropped_f: xuv.Ef_prop_cropped})
    out_ir = sess.run(padded_ir_f, feed_dict={ir_cropped_f: ir.Ef_prop_cropped})
    out_ir_time = sess.run(ir_time_domain, feed_dict={ir_cropped_f: ir.Ef_prop_cropped})

    plot_reconstructions(xuv, out_xuv, out_xuv_time)
    plot_reconstructions(ir, out_ir, out_ir_time)


def plot_reconstructions(field, out_f, out_time):

    # plotting
    fig = plt.figure()
    gs = fig.add_gridspec(4, 2)
    # plot the input
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(field.f_cropped, np.real(field.Ef_prop_cropped), color='purple', label='input')
    ax.plot(field.f_cropped, np.imag(field.Ef_prop_cropped), color='purple', alpha=0.5)
    ax.plot(field.fmat, np.zeros_like(field.fmat), color='black', alpha=0.5)
    ax.legend(loc=3)
    # plot the reconstruced complete xuv in frequency domain
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(field.fmat, np.real(field.Ef_prop), label='actual', color='orange')
    ax.plot(field.fmat, np.imag(field.Ef_prop), alpha=0.5, color='orange')
    ax.plot(field.fmat, np.real(out_f), label='padded', linestyle='dashed', color='black')
    ax.plot(field.fmat, np.imag(out_f), alpha=0.5, linestyle='dashed', color='black')
    ax.legend(loc=3)
    # plot the actual full xuv spectrum in frequency domain
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(field.fmat, np.real(field.Ef_prop), label='actual', color='orange')
    ax.plot(field.fmat, np.imag(field.Ef_prop), alpha=0.5, color='orange')
    ax.legend(loc=3)
    # tensorflow fourier transformed xuv in time
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(field.tmat, np.real(out_time), color='blue', label='tf fft of reconstruced')
    # ax.plot(field.tmat, np.imag(out_time), color='blue', alpha=0.5)
    # plot numpy fft of the reconstruced
    fft_rec = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(out_f)))
    ax.plot(field.tmat, np.real(fft_rec), color='black', linestyle='dashed', label='numpy fft of padded')
    ax.legend(loc=3)
    # plot the actual field in time
    ax = fig.add_subplot(gs[2,1])
    ax.plot(field.tmat, np.real(field.Et_prop), color='orange', label='actual')
    ax.legend(loc=3)
    # compare the tensorflow ifft and the actual
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(field.tmat, np.real(field.Et_prop), color='orange', label='actual')
    ax.plot(field.tmat, np.real(out_time), color='black', label='tf fft of reconstruced', linestyle='dashed')
    ax.legend(loc=3)


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


def plot_xuv_ir_trace():

    out_xuv_time = sess.run(xuv_time_domain, feed_dict={xuv_cropped_f: xuv.Ef_prop_cropped})

    out_ir_time = sess.run(ir_time_domain, feed_dict={ir_cropped_f: ir.Ef_prop_cropped})

    # trace
    trace = sess.run(image, feed_dict={xuv_cropped_f: xuv.Ef_prop_cropped,
                                     ir_cropped_f: ir.Ef_prop_cropped})

    fig = plt.figure()
    gs = fig.add_gridspec(3, 2)

    # plot the ir in time
    ax = fig.add_subplot(gs[0, :])
    ax.plot(ir.tmat, np.real(out_ir_time), color='blue')

    # plot the xuv in time
    ax = fig.add_subplot(gs[1, :])
    ax.plot(xuv.tmat, np.real(out_xuv_time), color='blue')

    # plot the trace
    ax = fig.add_subplot(gs[2, :])
    ax.pcolormesh(trace, cmap='jet')

    plt.savefig('./ir6phase')


def check_integrals():

    fig = plt.figure()
    gs = fig.add_gridspec(2,2)


    # plot A integral
    ax = fig.add_subplot(gs[0,:])
    out = sess.run(A_t_integ_t_phase, feed_dict={ir_cropped_f: ir.Ef_prop_cropped})
    ax.plot(np.real(out[256, :]))


    ax = fig.add_subplot(gs[1, :])
    out = sess.run(ir_time_domain, feed_dict={ir_cropped_f: ir.Ef_prop_cropped})
    ax.plot(np.real(out))


def plot_streaking_trace():
    out = sess.run(image, feed_dict={xuv_cropped_f: xuv.Ef_prop_cropped,
                                     ir_cropped_f: ir.Ef_prop_cropped})

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[:, :])
    ax.pcolormesh(out, cmap='jet')
    plt.savefig('./2.png')


def view_final_image():

    image_out = sess.run(image, feed_dict={ir_cropped_f: ir.Ef_prop_cropped, xuv_cropped_f: xuv.Ef_prop_cropped})

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[:, :])
    ax.pcolormesh(tau_values, p_values, image_out, cmap='jet')
    ax.text(0.0, 0.9, 'IR:\nI: {}\nconst phase: {} rad\nclambda: {} um\npulse duration: {} fs'.format(round(ir.I0_sc, 2), round(ir.const_phase, 2),
                                                                     round(ir.clambda, 2),
                                                                     round(ir.pulse_duration, 2)),
            transform=ax.transAxes, backgroundcolor='white')
    plt.savefig('./stuff/{}.png'.format(int(ir.const_phase)))


def build_graph(xuv_cropped_f_in, ir_cropped_f_in):

    global p
    global tau_index
    global tau_values
    global p_values
    global padded_xuv_f
    global xuv_time_domain
    global padded_ir_f
    global ir_time_domain




    # define constants
    xuv_fmat = tf.constant(xuv.fmat, dtype=tf.float32)
    ir_fmat = tf.constant(ir.fmat, dtype=tf.float32)

    # zero pad the spectrum of ir and xuv input to match the full fmat
    # [pad_before , padafter]
    paddings_xuv = tf.constant([[xuv.fmin_index, len(xuv.Ef_prop) - xuv.fmax_index]], dtype=tf.int32)
    padded_xuv_f = tf.pad(xuv_cropped_f_in, paddings_xuv)

    # same for the IR
    paddings_ir = tf.constant([[ir.fmin_index, len(ir.Ef_prop) - ir.fmax_index]], dtype=tf.int32)
    padded_ir_f = tf.pad(ir_cropped_f_in, paddings_ir)

    # fourier transform the padded xuv
    xuv_time_domain = tf_1d_ifft(tensor=padded_xuv_f, shift=int(len(xuv.fmat) / 2))

    # fourier transform the padded ir
    ir_time_domain = tf_1d_ifft(tensor=padded_ir_f, shift=int(len(ir.fmat) / 2))

    # zero pad the ir in frequency space to match dt of xuv
    assert (1 / (ir.df * xuv.dt)) - math.ceil((1 / (ir.df * xuv.dt))) < 0.000000001
    N_new = math.ceil(1 / (ir.df * xuv.dt))
    f_pad_2 = ir.df * np.arange(-N_new / 2, N_new / 2, 1)
    t_pad_2 = xuv.dt * np.arange(-N_new / 2, N_new / 2, 1)
    N_current = len(ir.fmat)
    pad_2 = (N_new - N_current) / 2
    assert int(pad_2) - pad_2 == 0
    paddings_ir_2 = tf.constant([[int(pad_2), int(pad_2)]], dtype=tf.int32)
    padded_ir_2 = tf.pad(padded_ir_f, paddings_ir_2)

    # calculate ir with matching dt in time
    ir_t_matched_dt = tf_1d_ifft(tensor=padded_ir_2, shift=int(N_new / 2))

    # match the scale of the original
    scale_factor = tf.constant(N_new / len(ir.Ef_prop), dtype=tf.complex64)

    ir_t_matched_dt_scaled = ir_t_matched_dt * scale_factor

    # integrate ir pulse
    A_t = tf.constant(-1.0 * xuv.dt, dtype=tf.float32) * tf.cumsum(tf.real(ir_t_matched_dt_scaled))
    flipped1 = tf.reverse(A_t, axis=[0])
    flipped_integral = tf.constant(-1.0 * xuv.dt, dtype=tf.float32) * tf.cumsum(flipped1, axis=0)
    A_t_integ_t_phase = tf.reverse(flipped_integral, axis=[0])

    # find middle index point
    middle = int(N_new / 2)
    rangevals = np.array(range(len(xuv.tmat))) - len(xuv.tmat) / 2
    middle_indexes = np.array([middle] * len(xuv.tmat)) + rangevals

    # maximum add to zero before would be out of bounds
    max_steps = int(N_new / 2 - len(xuv.tmat) / 2)

    # use this dt to scale the image size along tau axis
    dtau_index = 84 # to match measured
    # dtau_index = 75

    N_tau = int(max_steps / dtau_index)
    N_tau = 40

    if N_tau % 2 != 0:
        N_tau += -1

    tau_index = dtau_index * np.arange(-N_tau, N_tau, 1, dtype=int)

    # Number of points must be even
    assert N_tau % 2 == 0
    assert type(dtau_index) == int
    assert abs(tau_index[0]) < max_steps

    indexes = middle_indexes.reshape(-1, 1) + tau_index.reshape(1, -1)
    tau_values = tau_index * xuv.dt  # atomic units

    # gather values from integrated array
    ir_values = tf.gather(A_t_integ_t_phase, indexes.astype(np.int))
    ir_values = tf.expand_dims(ir_values, axis=0)

    # create momentum vector
    # p = np.linspace(3, 6.5, 200).reshape(-1, 1, 1) # previously
    p = np.linspace(1.917, 5.0719, 200).reshape(-1, 1, 1)
    p_values = np.squeeze(p)  # atomic units
    K = (0.5 * p ** 2)

    # convert to tensorflow
    p_tf = tf.constant(p, dtype=tf.float32)
    K_tf = tf.constant(K, dtype=tf.float32)

    # 3d ir mat
    p_A_t_integ_t_phase3d = p_tf * ir_values
    ir_phi = tf.exp(tf.complex(imag=(p_A_t_integ_t_phase3d), real=tf.zeros_like(p_A_t_integ_t_phase3d)))

    # add fourier transform term
    e_fft = np.exp(-1j * (K + med.Ip) * xuv.tmat.reshape(1, -1, 1))
    e_fft_tf = tf.constant(e_fft, dtype=tf.complex64)

    # add xuv to integrate over
    xuv_time_domain_integrate = tf.reshape(xuv_time_domain, [1, -1, 1])

    # multiply elements together
    product = xuv_time_domain_integrate * ir_phi * e_fft_tf

    # integrate over the xuv time
    integration = tf.constant(xuv.dt, dtype=tf.complex64) * tf.reduce_sum(product, axis=1)

    # absolute square the matrix
    image_not_scaled = tf.square(tf.abs(integration))

    scaled = image_not_scaled - tf.reduce_min(image_not_scaled)
    image = scaled / tf.reduce_max(scaled)

    return image


def remove_linear_phase(xuv_f, plotting=False):

    if plotting:
        plt.figure(33)
        plt.plot(np.real(xuv_f), color='blue')
        plt.plot(np.imag(xuv_f), color='red')

    # move intensity peak to center
    Et_prop_before_centered = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(xuv_f)))
    # find intensity peak
    center_index = int(len(Et_prop_before_centered) / 2)

    intensity_peak_index = np.argmin(np.abs(np.max(np.abs(Et_prop_before_centered)) - np.abs(Et_prop_before_centered)))
    if plotting:
        plt.figure(34)
        plt.plot(np.abs(Et_prop_before_centered))
        plt.plot([center_index, center_index], [-0.025, 0.025])
        plt.plot([intensity_peak_index, intensity_peak_index], [-0.025, 0.025], color='blue')

    roll_num = center_index - intensity_peak_index
    if roll_num > 0:
        Et_prop_before_centered[-roll_num:] = 0.0
    if roll_num < 0:
        Et_prop_before_centered[:-roll_num] = 0.0

    rolled_Et = np.roll(Et_prop_before_centered, roll_num)

    if plotting:
        plt.figure(35)
        plt.plot(np.abs(rolled_Et))
        plt.plot([center_index, center_index], [-0.025, 0.025])
        plt.plot([intensity_peak_index, intensity_peak_index], [-0.025, 0.025], color='blue')

    # reverse foutier transform again
    xuv_f_adjusted = np.fft.fftshift(np.fft.fft(np.fft.fftshift(rolled_Et)))

    if plotting:
        plt.figure(36)
        plt.plot(np.real(xuv_f_adjusted), color='blue')
        plt.plot(np.imag(xuv_f_adjusted), color='red')

    return xuv_f_adjusted




# retrieve xuv spectrum
with open('measured_spectrum.p', 'rb') as file:
    spectrum_data = pickle.load(file)

# DEFINE THE XUV FIELD
# transform limited
# xuv = XUV_Field()

# specific phase
# xuv = XUV_Field(gdd=500.0, tod=0.0)

# random xuv
# xuv = XUV_Field(random_phase={'nodes': 100, 'amplitude': 6})

# random xuv with measured spectrum
# xuv = XUV_Field(random_phase={'nodes': 100, 'amplitude': 6}, measured_spectrum=spectrum_data)


xuv = XUV_Field(random_phase_taylor={'coefs': 3, 'amplitude': 200},
                measured_spectrum=spectrum_data)

# xuv = XUV_Field(random_phase_taylor={'coefs': 3, 'amplitude': 200})






## DEFINE THE IR FIELD
# default ir field
ir = IR_Field()

# specific ir field
# ir = IR_Field(const_phase=0.0, pulse_duration=10.0, clambda=1.7)


# random ir field
# ir = IR_Field(random_pulse={'phase_range':(0,2*np.pi), 'clambda_range': (1.2,2.3), 'pulse_duration_range':(7.0,12.0),
#                             'I_range':(0.5,1.0)})



# the length of each vector, ir and xuv for plotting
xuv_frequency_grid_length = xuv.fmax_index - xuv.fmin_index
ir_frequency_grid_length = ir.fmax_index - ir.fmin_index

med = Med()

# construct the field with tensorflow

# placeholders
xuv_cropped_f = tf.placeholder(tf.complex64, [len(xuv.Ef_prop_cropped)])
ir_cropped_f = tf.placeholder(tf.complex64, [len(ir.Ef_prop_cropped)])


# build the tf graph with the inputs
image = build_graph(xuv_cropped_f_in=xuv_cropped_f, ir_cropped_f_in=ir_cropped_f)






if __name__ == "__main__":


    # plot the xuv field
    plot_initial_field(field=xuv, timespan=int(xuv_frequency_grid_length))

    # plot the infrared field
    plot_initial_field(field=ir, timespan=int(ir_frequency_grid_length))


    # image size
    print('N p : ', len(p))
    print('N tau : ', len(tau_index))
    tau_values_si = tau_values * sc.physical_constants['atomic unit of time'][0] * 1e18
    print('XUV dt: ', (tau_values_si[-1] - tau_values_si[-2]), 'attoseconds')

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()

        view_final_image()

        check_fft_and_reconstruction()

        # check_padded_time_domain()

        # check_corner_errors()

        # plot_xuv_ir_trace()

        # check_integrals()

        # plot_streaking_trace()


        plt.show()
