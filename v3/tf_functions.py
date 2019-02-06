import tensorflow as tf
import xuv_spectrum.spectrum
import ir_spectrum.ir_spectrum
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial
import scipy.constants as sc

# print(xuv_spectrum.spectrum.params.keys())
# exit(0)

# plt.figure(1)
# plt.plot(xuv_spectrum.spectrum.params["Ef"])
# plt.show()




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






def xuv_taylor_to_E(coef_values_normalized, coefs, amplitude):
    # print(coef_values_normalized)

    # eventually, will have to convert everything to atomic units before inputting here!!
    Ef = tf.constant(xuv_spectrum.spectrum.params["Ef"], dtype=tf.complex64)
    Ef = tf.reshape(Ef, [1, -1])

    fmat_taylor = tf.constant(xuv_spectrum.spectrum.params["fmat"]-xuv_spectrum.spectrum.params["f0"], dtype=tf.float32)

    # create factorials
    factorials = tf.constant(factorial(np.array(range(coefs))+1), dtype=tf.float32)
    factorials = tf.reshape(factorials, [1, -1, 1])

    # create exponents
    exponents = tf.constant(np.array(range(coefs))+1, dtype=tf.float32)

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
    coef_values = tf.reshape(coef_values_normalized, [tf.shape(coef_values_normalized)[0], -1, 1]) * amplitude_scaler

    # divide by the factorials
    coef_div_fact = tf.divide(coef_values, factorials)

    # multiply by the fmat
    taylor_coefs_mat = coef_div_fact * exp_mat_fmat

    # this is the phase angle, summed along the taylor terms
    taylor_terms_summed = tf.reduce_sum(taylor_coefs_mat, axis=1)

    # apply the phase angle to Ef
    Ef_prop = Ef * tf.exp(tf.complex(imag=taylor_terms_summed, real=tf.zeros_like(taylor_terms_summed)))


    # set phase angel to 0!!!!!
    # return Ef_prop


def ir_from_params(ir_param_values, amplitudes):

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
    Up = (sc.elementary_charge ** 2 * scaled_tf_values_si["I"]) / (2 * sc.c * sc.epsilon_0 * sc.electron_mass * (2 * np.pi * scaled_tf_values_si["f0"]) ** 2)

    # convert to AU
    values_au = {}
    values_au["Up"] = Up / sc.physical_constants['atomic unit of energy'][0]
    values_au["f0"] = scaled_tf_values_si["f0"] * sc.physical_constants['atomic unit of time'][0]
    values_au["t0"] = scaled_tf_values_si["t0"] / sc.physical_constants['atomic unit of time'][0]

    # calculate driving amplitude in AU
    E0 = tf.sqrt(4 * values_au["Up"] * (2 * np.pi * values_au["f0"]) ** 2)

    # set up the driving IR field amplitude in AU
    tf_tmat = tf.reshape(tf.constant(ir_spectrum.ir_spectrum.tmat, dtype=tf.float32), [1, -1])
    tf_fmat = tf.reshape(tf.constant(ir_spectrum.ir_spectrum.fmat, dtype=tf.float32), [1, -1])

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



    with tf.Session() as sess:
        input_values = np.array([[0.0, 0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.0, 0.0]])

        out = sess.run(Et_phase, feed_dict={ir_values_in: input_values})
        print(np.shape(out))
        plt.figure(1)
        plt.plot(np.real(out[0]))
        plt.figure(2)
        plt.plot(np.real(out[1]))
        plt.show()

    exit(0)

    Up = (sc.elementary_charge ** 2 * I0) / (2 * sc.c * sc.epsilon_0 * sc.electron_mass * (2 * np.pi * f0) ** 2)
    E0 = np.sqrt(4 * Up * (2 * np.pi * f0) ** 2)
    Et = E0 * np.exp(-2 * np.log(2) * (ir_spectrum.ir_spectrum.tmat / t0) ** 2) * np.exp(1j * 2 * np.pi * f0 * ir_spectrum.ir_spectrum.tmat)



    exit(0)





coefs_in = tf.placeholder(tf.float32, shape=[None, 5])
Ef_prop = xuv_taylor_to_E(coefs_in, coefs=5, amplitude=12.0)

# ir amplitudes
amplitudes = {}
amplitudes["phase_range"] = (0, 2 * np.pi)
# amplitudes["clambda_range"] = (1.5, 1.6345)
amplitudes["clambda_range"] = (1.0, 1.6345)
amplitudes["pulseduration_range"] =  (7.0, 12.0)
amplitudes["I_range"] = (0.4, 1.0)
ir_values_in = tf.placeholder(tf.float32, shape=[None, 4])
ir = ir_from_params(ir_values_in, amplitudes=amplitudes)



exit(0)

with tf.Session() as sess:
        # input_array = np.array([[0.01, 0.02, 0.03, 0.04, 0.05],
        #                         [0.12, 0.1, 0.2, 0.13, 0.16]])
        input_array = np.array([[0.00, 0.5, 0.00, 0.00, 0.00],
                                [0.00, -0.5, 0.00, 0.00, 0.00]])
        # print(np.shape(sess.run(taylor_terms_summed, feed_dict={coefs_in: input_array})))
        out = sess.run(Ef_prop, feed_dict={coefs_in: input_array})

        plt.figure(1)
        plt.plot(np.real(out[0]), color="blue")
        plt.plot(np.imag(out[0]), color='red')
        plt.plot(np.abs(out[0]), color='black')
        axtwin = plt.gca().twinx()
        axtwin.plot(np.unwrap(np.angle(out[0])), color='green')

        plt.figure(2)
        plt.plot(np.real(out[1]), color="blue")
        plt.plot(np.imag(out[1]), color='red')
        plt.plot(np.abs(out[1]), color='black')
        axtwin = plt.gca().twinx()
        axtwin.plot(np.unwrap(np.angle(out[1])), color='green')

        plt.show()
        print(out)
        exit(0)



# ir_construct =
