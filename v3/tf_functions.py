import tensorflow as tf
import xuv_spectrum.spectrum
import ir_spectrum.ir_spectrum
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial
import scipy.constants as sc
import math
import phase_parameters.params
import generate_data3



def tf_ifft(tensor, shift, axis=0):

    shifted = tf.manip.roll(tensor, shift=shift, axis=axis)
    # fft
    time_domain_not_shifted = tf.ifft(shifted)
    # shift again
    time_domain = tf.manip.roll(time_domain_not_shifted, shift=shift, axis=axis)

    return time_domain


def tf_fft(tensor, shift, axis=0):

    shifted = tf.manip.roll(tensor, shift=shift, axis=axis)
    # fft
    freq_domain_not_shifted = tf.fft(shifted)
    # shift again
    freq_domain = tf.manip.roll(freq_domain_not_shifted, shift=shift, axis=axis)

    return freq_domain


def xuv_taylor_to_E(coefficients_in):

    assert int(coefficients_in.shape[1]) == phase_parameters.params.xuv_phase_coefs

    amplitude = phase_parameters.params.amplitude

    Ef = tf.constant(xuv_spectrum.spectrum.Ef, dtype=tf.complex64)
    Ef = tf.reshape(Ef, [1, -1])

    fmat_taylor = tf.constant(xuv_spectrum.spectrum.fmat-xuv_spectrum.spectrum.f0, dtype=tf.float32)

    # create factorials
    factorials = tf.constant(factorial(np.array(range(coefficients_in.shape[1]))+1), dtype=tf.float32)
    factorials = tf.reshape(factorials, [1, -1, 1])

    # create exponents
    exponents = tf.constant(np.array(range(coefficients_in.shape[1]))+1, dtype=tf.float32)

    # reshape the taylor fmat
    fmat_taylor = tf.reshape(fmat_taylor, [1, 1, -1])

    # reshape the exponential matrix
    exp_mat = tf.reshape(exponents, [1, -1, 1])

    # raise the fmat to the exponential power
    exp_mat_fmat = tf.pow(fmat_taylor, exp_mat)

    # scale the coefficients
    amplitude_mat = tf.constant(amplitude, dtype=tf.float32)
    amplitude_mat = tf.reshape(amplitude_mat, [1, -1, 1])

    # amplitude scales with exponent
    amplitude_scaler = tf.pow(amplitude_mat, exp_mat)

    # reshape the coef values and scale them
    coef_values = tf.reshape(coefficients_in, [tf.shape(coefficients_in)[0], -1, 1]) * amplitude_scaler

    # divide by the factorials
    coef_div_fact = tf.divide(coef_values, factorials)

    # multiply by the fmat
    taylor_coefs_mat = coef_div_fact * exp_mat_fmat

    # this is the phase angle, summed along the taylor terms
    taylor_terms_summed = tf.reduce_sum(taylor_coefs_mat, axis=1)

    # apply the phase angle to Ef
    Ef_prop = Ef * tf.exp(tf.complex(imag=taylor_terms_summed, real=tf.zeros_like(taylor_terms_summed)))

    # fourier transform for time propagated signal
    Et_prop = tf_ifft(Ef_prop, shift=int(xuv_spectrum.spectrum.N/2), axis=1)

    # return the cropped E
    Ef_prop_cropped = Ef_prop[:, xuv_spectrum.spectrum.indexmin: xuv_spectrum.spectrum.indexmax]

    E_prop = {}
    E_prop["f"] = Ef_prop
    E_prop["f_cropped"] = Ef_prop_cropped
    E_prop["t"] = Et_prop
    #E_prop["coefs_divided_by_int"] = coefs_divided_by_int

    return E_prop



def ir_from_params(ir_param_values):

    amplitudes = phase_parameters.params.ir_param_amplitudes

    # construct tf nodes for middle and half range of inputs
    parameters = {}
    for key in ["phase_range", "clambda_range", "pulseduration_range", "I_range"]:
        parameters[key] = {}

        # get the middle and half the range of the variables
        parameters[key]["avg"] = (amplitudes[key][0] + amplitudes[key][1])/2
        parameters[key]["half_range"] = (amplitudes[key][1] - amplitudes[key][0]) / 2

        # create tensorflow constants
        parameters[key]["tf_avg"] = tf.constant(parameters[key]["avg"], dtype=tf.float32)
        parameters[key]["tf_half_range"] = tf.constant(parameters[key]["half_range"], dtype=tf.float32)


    # construct param values from normalized input
    scaled_tf_values = {}

    for i, key in enumerate(["phase_range", "clambda_range", "pulseduration_range", "I_range"]):
        scaled_tf_values[key.split("_")[0]] = parameters[key]["tf_avg"] + ir_param_values[:, i] * parameters[key]["tf_half_range"]

    # convert to SI units
    W = 1
    cm = 1e-2
    um = 1e-6
    fs = 1e-15
    atts = 1e-18

    scaled_tf_values_si = {}
    scaled_tf_values_si["I"] = scaled_tf_values["I"] * 1e13 * W / cm ** 2
    scaled_tf_values_si["f0"] = sc.c / (um * scaled_tf_values["clambda"])
    scaled_tf_values_si["t0"] =  scaled_tf_values["pulseduration"] * fs

    # calculate ponderomotive energy in SI units
    Up = (sc.elementary_charge ** 2 * tf.abs(scaled_tf_values_si["I"])) / (2 * sc.c * sc.epsilon_0 * sc.electron_mass * (2 * np.pi * scaled_tf_values_si["f0"]) ** 2)

    # convert to AU
    values_au = {}
    values_au["Up"] = Up / sc.physical_constants['atomic unit of energy'][0]
    values_au["f0"] = scaled_tf_values_si["f0"] * sc.physical_constants['atomic unit of time'][0]
    values_au["t0"] = scaled_tf_values_si["t0"] / sc.physical_constants['atomic unit of time'][0]

    # calculate driving amplitude in AU
    E0 = tf.sqrt(4 * values_au["Up"] * (2 * np.pi * values_au["f0"]) ** 2)

    # set up the driving IR field amplitude in AU
    tf_tmat = tf.reshape(tf.constant(ir_spectrum.ir_spectrum.tmat, dtype=tf.float32), [1, -1])
    # tf_fmat = tf.reshape(tf.constant(ir_spectrum.ir_spectrum.fmat, dtype=tf.float32), [1, -1])

    # slow oscilating envelope
    Et_slow_osc = tf.reshape(E0, [-1, 1]) * tf.exp(-2*np.log(2) * (tf_tmat / tf.reshape(values_au["t0"], [-1, 1]))**2)

    # fast oscilating envelope
    phase = 2 * np.pi * tf.reshape(values_au["f0"], [-1, 1]) * tf_tmat
    Et_fast_osc = tf.exp(tf.complex(imag=phase, real=tf.zeros_like(phase)))

    # Pulse before phase applied
    Et = tf.complex(real=Et_slow_osc, imag=tf.zeros_like(Et_slow_osc)) * Et_fast_osc

    # Fourier transform
    Ef = tf_fft(Et, shift=int(len(ir_spectrum.ir_spectrum.tmat)/2), axis=1)

    # apply phase angle
    phase = tf.reshape(scaled_tf_values["phase"], [-1, 1])
    Ef_phase = Ef * tf.exp(tf.complex(imag=phase, real=tf.zeros_like(phase)))

    # inverse fourier transform
    Et_phase = tf_ifft(Ef_phase, shift=int(len(ir_spectrum.ir_spectrum.tmat) / 2), axis=1)

    # crop the phase
    Ef_phase_cropped = Ef_phase[:, ir_spectrum.ir_spectrum.start_index:ir_spectrum.ir_spectrum.end_index]

    E_prop = {}
    E_prop["f"] = Ef_phase
    E_prop["f_cropped"] = Ef_phase_cropped
    E_prop["t"] = Et_phase

    return E_prop


def streaking_trace(xuv_cropped_f_in, ir_cropped_f_in):

    Ip = phase_parameters.params.Ip

    global p
    global tau_index
    global tau_values
    global p_values
    global k_values
    global padded_xuv_f
    global xuv_time_domain
    global padded_ir_f
    global ir_time_domain
    global eV_values




    # define constants
    # xuv_fmat = tf.constant(xuv.fmat, dtype=tf.float32)
    # ir_fmat = tf.constant(ir.fmat, dtype=tf.float32)

    # zero pad the spectrum of ir and xuv input to match the full fmat
    # [pad_before , padafter]
    paddings_xuv = tf.constant([[xuv_spectrum.spectrum.indexmin, xuv_spectrum.spectrum.N - xuv_spectrum.spectrum.indexmax]], dtype=tf.int32)
    padded_xuv_f = tf.pad(xuv_cropped_f_in, paddings_xuv)

    # same for the IR
    paddings_ir = tf.constant([[ir_spectrum.ir_spectrum.start_index, ir_spectrum.ir_spectrum.N - ir_spectrum.ir_spectrum.end_index]], dtype=tf.int32)
    padded_ir_f = tf.pad(ir_cropped_f_in, paddings_ir)

    # fourier transform the padded xuv
    xuv_time_domain = tf_ifft(tensor=padded_xuv_f, shift=int(xuv_spectrum.spectrum.N / 2))

    # fourier transform the padded ir
    ir_time_domain = tf_ifft(tensor=padded_ir_f, shift=int(ir_spectrum.ir_spectrum.N / 2))

    # zero pad the ir in frequency space to match dt of xuv
    assert (1 / (ir_spectrum.ir_spectrum.df * xuv_spectrum.spectrum.dt)) - math.ceil((1 / (ir_spectrum.ir_spectrum.df * xuv_spectrum.spectrum.dt))) < 0.000000001
    N_new = math.ceil(1 / (ir_spectrum.ir_spectrum.df * xuv_spectrum.spectrum.dt))
    f_pad_2 = ir_spectrum.ir_spectrum.df * np.arange(-N_new / 2, N_new / 2, 1)
    t_pad_2 = xuv_spectrum.spectrum.dt * np.arange(-N_new / 2, N_new / 2, 1)
    N_current = ir_spectrum.ir_spectrum.N
    pad_2 = (N_new - N_current) / 2
    assert int(pad_2) - pad_2 == 0
    paddings_ir_2 = tf.constant([[int(pad_2), int(pad_2)]], dtype=tf.int32)
    padded_ir_2 = tf.pad(padded_ir_f, paddings_ir_2)

    # calculate ir with matching dt in time
    ir_t_matched_dt = tf_ifft(tensor=padded_ir_2, shift=int(N_new / 2))

    # match the scale of the original
    scale_factor = tf.constant(N_new / ir_spectrum.ir_spectrum.N, dtype=tf.complex64)

    ir_t_matched_dt_scaled = ir_t_matched_dt * scale_factor

    # integrate ir pulse
    A_t = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(tf.real(ir_t_matched_dt_scaled))
    flipped1 = tf.reverse(A_t, axis=[0])
    flipped_integral = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(flipped1, axis=0)
    A_t_integ_t_phase = tf.reverse(flipped_integral, axis=[0])

    # find middle index point
    middle = int(N_new / 2)
    rangevals = np.array(range(xuv_spectrum.spectrum.N)) - xuv_spectrum.spectrum.N / 2
    middle_indexes = np.array([middle] * xuv_spectrum.spectrum.N) + rangevals

    # maximum add to zero before would be out of bounds
    max_steps = int(N_new / 2 - xuv_spectrum.spectrum.N / 2)

    # use this dt to scale the image size along tau axis
    #dtau_index = 84  # to match measured
    dtau_index = 180 # to match measured
    # dtau_index = 75

    N_tau = int(max_steps / dtau_index)


    if N_tau % 2 != 0:
        N_tau += -1

    N_tau = 29

    tau_index = dtau_index * np.arange(-N_tau, N_tau, 1, dtype=int)

    # Number of points must be even
    # assert N_tau % 2 == 0
    # assert type(dtau_index) == int
    # assert abs(tau_index[0]) < max_steps

    indexes = middle_indexes.reshape(-1, 1) + tau_index.reshape(1, -1)
    tau_values = tau_index * xuv_spectrum.spectrum.dt  # atomic units

    #print(tau_values*sc.physical_constants['atomic unit of time'][0])
    #exit(0)

    # gather values from integrated array
    ir_values = tf.gather(A_t_integ_t_phase, indexes.astype(np.int))
    ir_values = tf.expand_dims(ir_values, axis=0)

    # create momentum vector
    #p = np.linspace(3, 6.5, 200).reshape(-1, 1, 1) # previously
    #p = np.linspace(1.917, 5.0719, 200).reshape(-1, 1, 1)
    # p = np.linspace(1.8, 5.5, 235).reshape(-1, 1, 1)
    # p_values = np.squeeze(p)  # atomic units
    # K = (0.5 * p ** 2)

    # set dK = 1eV
    K = np.arange(50, 351, 1) # eV
    eV_values = np.array(K)
    # convert K to atomic units
    K = K * sc.electron_volt  # joules
    K = K / sc.physical_constants['atomic unit of energy'][0]  # a.u.
    K = K.reshape(-1, 1, 1)
    p = np.sqrt(2 * K).reshape(-1, 1, 1)
    k_values = np.squeeze(K) # atomic untis
    p_values = np.squeeze(p)  # atomic units

    # convert to tensorflow
    p_tf = tf.constant(p, dtype=tf.float32)
    K_tf = tf.constant(K, dtype=tf.float32)

    # 3d ir mat
    p_A_t_integ_t_phase3d = p_tf * ir_values
    ir_phi = tf.exp(tf.complex(imag=(p_A_t_integ_t_phase3d), real=tf.zeros_like(p_A_t_integ_t_phase3d)))

    # add fourier transform term
    e_fft = np.exp(-1j * (K + Ip) * xuv_spectrum.spectrum.tmat.reshape(1, -1, 1))
    e_fft_tf = tf.constant(e_fft, dtype=tf.complex64)

    # add xuv to integrate over
    xuv_time_domain_integrate = tf.reshape(xuv_time_domain, [1, -1, 1])

    # multiply elements together
    product = xuv_time_domain_integrate * ir_phi * e_fft_tf

    # integrate over the xuv time
    integration = tf.constant(xuv_spectrum.spectrum.dt, dtype=tf.complex64) * tf.reduce_sum(product, axis=1)

    # absolute square the matrix
    image_not_scaled = tf.square(tf.abs(integration))

    scaled = image_not_scaled - tf.reduce_min(image_not_scaled)
    image = scaled / tf.reduce_max(scaled)

    parameters = {}
    parameters["p"] = p
    parameters["tau_index"] = tau_index
    parameters["tau_values"] = tau_values
    parameters["p_values"] = p_values
    parameters["k_values"] = k_values
    parameters["padded_xuv_f"] = padded_xuv_f
    parameters["xuv_time_domain"] = xuv_time_domain
    parameters["padded_ir_f"] = padded_ir_f
    parameters["ir_time_domain"] = ir_time_domain
    parameters["eV_values"] = eV_values

    return image, parameters


if __name__ == "__main__":

    # xuv creation
    xuv_coefs_in = tf.placeholder(tf.float32, shape=[None, phase_parameters.params.xuv_phase_coefs])
    xuv_E_prop = xuv_taylor_to_E(xuv_coefs_in)

    # IR creation
    # ir amplitudes
    amplitudes = {}
    amplitudes["phase_range"] = (0, 2 * np.pi)
    # amplitudes["clambda_range"] = (1.6345, 1.6345)
    amplitudes["clambda_range"] = (1.0, 1.6345)
    amplitudes["pulseduration_range"] =  (7.0, 12.0)
    amplitudes["I_range"] = (0.4, 1.0)


    ir_values_in = tf.placeholder(tf.float32, shape=[None, 4])
    ir_E_prop = ir_from_params(ir_values_in)


    # construct streaking image
    image, _ = streaking_trace(xuv_cropped_f_in=xuv_E_prop["f_cropped"][0], ir_cropped_f_in=ir_E_prop["f_cropped"][0])




    with tf.Session() as sess:

        # count the number of bad samples generated
        bad_samples = 0

        # xuv_input = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]])
        indexmin = 100
        indexmax = (2*1024) - 100
        xuv_input = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        xuv_t = sess.run(xuv_E_prop["t"], feed_dict={xuv_coefs_in: xuv_input})
        threshold = np.max(np.abs(xuv_t)) / 300


        # xuv_coefs_rand = (2 * np.random.rand(4) - 1.0).reshape(1, -1)
        xuv_coefs_rand = np.array([[0.0, 1.0, 1.0, 1.0]])

        xuv_input = np.append(np.array([[0.0]]), xuv_coefs_rand, axis=1)
        # exit(0)
        print(sess.run(xuv_E_prop2["coefs_divided_by_int"], feed_dict={xuv_coefs_in: xuv_input}))
        xuv_t = sess.run(xuv_E_prop2["t"], feed_dict={xuv_coefs_in: xuv_input})

        fig = plt.figure()
        gs = fig.add_gridspec(2, 2)
        ax = fig.add_subplot(gs[:, :])
        ax.plot(np.real(xuv_t[0]), color="blue")
        ax.plot(np.imag(xuv_t[0]), color="red")
        plt.show()



        bad_samples, xuv_good = generate_data3.check_time_boundary(indexmin, indexmax, threshold, xuv_t[0], bad_samples)

        print(xuv_good)

        exit(0)





    with tf.Session() as sess:
        xuv_input = np.array([[0.0, 0.0, 0.0, 1.0, 0.0]])
        # xuv_input = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
        ir_input = np.array([[0.0, 0.0, 0.0, 0.0]])

        out = sess.run(xuv_E_prop["f_cropped"], feed_dict={xuv_coefs_in: xuv_input,
                                         ir_values_in: ir_input})
        plt.figure(1)
        plt.plot(np.real(out[0]), color='blue')
        plt.plot(np.imag(out[0]), color='red')

        out = sess.run(xuv_E_prop["t"], feed_dict={xuv_coefs_in: xuv_input,
                                                           ir_values_in: ir_input})
        plt.figure(2)
        plt.plot(np.real(out[0]), color='blue')
        plt.plot(np.imag(out[0]), color='red')


        out = sess.run(image, feed_dict={xuv_coefs_in: xuv_input,
                                                   ir_values_in: ir_input})
        plt.figure(3)
        plt.pcolormesh(out)
        # plt.savefig("./4.png")
        plt.show()

