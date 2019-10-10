import numpy as np
import scipy.constants as sc
import pickle
import os
import sys
import tensorflow as tf
current_path = os.path.dirname(__file__)
sys.path.append(os.path.join(current_path+".."))
import matplotlib.pyplot as plt
import measured_trace.get_trace as measured_trace
import tf_functions
import xuv_spectrum.spectrum


central_wavelength = measured_trace.lam0*1e6 # [um] micrometer
ir_param_amplitudes = {}
ir_param_amplitudes["phase_range"] = (0, 2 * np.pi)
# use the central IR wavelength from trace
ir_param_amplitudes["clambda_range"] = (central_wavelength, central_wavelength) # [um] micrometer
# ir_param_amplitudes["clambda_range"] = (1.0, 1.6345)
ir_param_amplitudes["pulseduration_range"] = (11.0, 16.0) # [fs] femtosecond
ir_param_amplitudes["I_range"] = (0.02, 0.12)


#---------------------------
#--STREAKING TRACE PARAMS---
#---------------------------
Ip_eV = 24.587 # eV
Ip = Ip_eV * sc.electron_volt  # joules
Ip = Ip / sc.physical_constants['atomic unit of energy'][0]  # a.u.
delay_values = measured_trace.delay # femtosecond
delay_values_fs = delay_values * 1e15
K = measured_trace.energy

# threshold scaler for the generated pulses
threshold_scaler = 0.03

threshold_min_index = 100
# threshold_min_index = 50
threshold_max_index = (2*1024) - 100
# threshold_max_index = 1024 - 50


# ir pulse parameters
ir_pulse = {}
# pulse params
ir_pulse["N"] = 128
ir_pulse["tmax"] = 50e-15
ir_pulse["start_index"] = 64
ir_pulse["end_index"] = 84


xuv_pulse = {}
xuv_pulse["N"] = int(2 * 1024)
xuv_pulse["tmax"] = 1600e-18
# this scaled the applied spectral phase
# these values have to be tuned to keep the xuv pulse within the time range
#                                             1st   2nd  3rd   4th   5th order
xuv_pulse["coef_phase_amplitude"] = np.array([0.0, 1.3, 0.15, 0.03, 0.01])
# includes linear
xuv_phase_coefs=5
# phase amplitude
amplitude=20.0


if __name__ == "__main__":

    xuv_coefs = tf.placeholder(tf.float32, shape=[None, 5])
    xuv_gen = tf_functions.xuv_taylor_to_E(xuv_coefs)

    with tf.Session() as sess:

        # 2nd order
        feed_dict = {xuv_coefs:np.array([[0.0,1.0,0.0,0.0,0.0]])}
        xuv_photon_out_2 = sess.run(xuv_gen["t_photon"], feed_dict=feed_dict)
        xuv_electron_out_2 = sess.run(xuv_gen["t"], feed_dict=feed_dict)

        # 3rd order
        feed_dict = {xuv_coefs:np.array([[0.0,0.0,1.0,0.0,0.0]])}
        xuv_photon_out_3 = sess.run(xuv_gen["t_photon"], feed_dict=feed_dict)
        xuv_electron_out_3 = sess.run(xuv_gen["t"], feed_dict=feed_dict)

        # 4th order
        feed_dict = {xuv_coefs:np.array([[0.0,0.0,0.0,1.0,0.0]])}
        xuv_photon_out_4 = sess.run(xuv_gen["t_photon"], feed_dict=feed_dict)
        xuv_electron_out_4 = sess.run(xuv_gen["t"], feed_dict=feed_dict)

        # 5th order
        feed_dict = {xuv_coefs:np.array([[0.0,0.0,0.0,0.0,1.0]])}
        xuv_photon_out_5 = sess.run(xuv_gen["t_photon"], feed_dict=feed_dict)
        xuv_electron_out_5 = sess.run(xuv_gen["t"], feed_dict=feed_dict)

    fig = plt.figure(figsize=(9,7))
    gs = fig.add_gridspec(2, 4)

    ax = fig.add_subplot(gs[0,0])
    ax.set_title("2nd order")
    ax.plot(xuv_spectrum.spectrum.tmat_as,np.real(xuv_electron_out_2[0]), label="electron", color="purple")
    ax.legend()
    ax = fig.add_subplot(gs[1,0])
    ax.set_title("2nd order")
    ax.plot(xuv_spectrum.spectrum.tmat_as,np.real(xuv_photon_out_2[0]), label="photon", color="blue")
    ax.legend()

    ax = fig.add_subplot(gs[0,1])
    ax.set_title("3nd order")
    ax.plot(xuv_spectrum.spectrum.tmat_as,np.real(xuv_electron_out_3[0]), label="electron", color="purple")
    ax.legend()
    ax = fig.add_subplot(gs[1,1])
    ax.set_title("3nd order")
    ax.plot(xuv_spectrum.spectrum.tmat_as,np.real(xuv_photon_out_3[0]), label="photon", color="blue")
    ax.legend()

    ax = fig.add_subplot(gs[0,2])
    ax.set_title("4nd order")
    ax.plot(xuv_spectrum.spectrum.tmat_as,np.real(xuv_electron_out_4[0]), label="electron", color="purple")
    ax.legend()
    ax = fig.add_subplot(gs[1,2])
    ax.set_title("4nd order")
    ax.plot(xuv_spectrum.spectrum.tmat_as,np.real(xuv_photon_out_4[0]), label="photon", color="blue")
    ax.legend()

    ax = fig.add_subplot(gs[0,3])
    ax.set_title("5nd order")
    ax.plot(xuv_spectrum.spectrum.tmat_as,np.real(xuv_electron_out_5[0]), label="electron", color="purple")
    ax.legend()
    ax = fig.add_subplot(gs[1,3])
    ax.set_title("5nd order")
    ax.plot(xuv_spectrum.spectrum.tmat_as,np.real(xuv_photon_out_5[0]), label="photon", color="blue")
    ax.legend()


    plt.show()





