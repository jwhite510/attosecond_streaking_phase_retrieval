import tensorflow as tf
import phase_parameters.params
import scipy.constants as sc
import tf_functions
import numpy as np
import matplotlib.pyplot as plt
import xuv_spectrum.spectrum as xuv_spectrum
import scipy.interpolate

def tf_ifft(tensor, shift, axis=0):

    shifted = tf.manip.roll(tensor, shift=shift, axis=axis)
    # fft
    time_domain_not_shifted = tf.ifft(shifted)
    # shift again
    time_domain = tf.manip.roll(time_domain_not_shifted, shift=shift, axis=axis)

    return time_domain

def  ev_to_au(vector):
    """
    ev_to_Hz ( vector [eV] )
    return vector [a.u (time)]
    """
    vector_joules = np.array(vector) * sc.electron_volt # joules
    # Energy[J] = h[J*s] f[1/s]
    # convert joules to Hz
    # sc.h [J*s]
    vector_hz = vector_joules / sc.h # Hz
    vector_au_time = vector_hz / sc.physical_constants['atomic unit of time'][0]  # a.u. 1/time
    return vector_au_time



def ev_to_p(vector):
    """
    ev_to_p(vector [eV])
    return vector [a.u. momenrum]
    """
    vector =  np.array(vector) * sc.electron_volt # joules
    vector = vector / sc.physical_constants['atomic unit of energy'][0]  # a.u. energy
    vector = np.sqrt(2 * vector) # a.u. momentum

    return vector


def au_energy_to_ev_energy(vector):
    """
    vector[a.u. energy]

    """
    vector = np.array(vector) * sc.physical_constants['atomic unit of energy'][0]  # joules energy
    vector = vector / sc.electron_volt # electron volts

    return vector



if __name__ == "__main__":

    # Fourier transform
    # xuv_spectrum.tmat
    # xuv_spectrum.dt
    xuv_Et_electron = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(xuv_spectrum.Ef)))
    xuv_Et_photon = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(xuv_spectrum.Ef_photon)))

    # plt.plot(np.real(xuv_Et_electron))
    # plt.plot(np.real(xuv_Et_photon))
    # plt.show()


    # momentum in atomic units
    K = phase_parameters.params.K * sc.electron_volt  # joules
    K = K / sc.physical_constants['atomic unit of energy'][0]  # a.u. energy
    K = K.reshape(-1, 1)
    p = np.sqrt(2 * K) # a.u. momentum
    # K2 = (p**2)/2


    # fourier transform term
    Ip = phase_parameters.params.Ip # a.u.
    # e_fft = np.exp(-1j * (K + Ip) * xuv_spectrum.tmat.reshape(1, -1))

    # e_fft = np.exp(-1j * (K) * xuv_spectrum.tmat.reshape(1, -1))
    # e_fft = np.exp(-1j*((p**2)/2)*xuv_spectrum.tmat)
    e_fft = np.exp(-1j*(    ((p**2)/2) + Ip )*xuv_spectrum.tmat)

    # create interpolator for cross section
    # hv = 0.5p^2 +Ip
    electron_energy_ev = xuv_spectrum.cross_section_ev-au_energy_to_ev_energy(Ip)
    electron_au_momentum = ev_to_p(electron_energy_ev)

    interpolator = scipy.interpolate.interp1d(electron_au_momentum, xuv_spectrum.cross_section, kind='linear')
    cross_section_p = interpolator(np.squeeze(p))


    cross_section_p_sqrt = np.sqrt(cross_section_p).reshape(-1, 1)

    # integrate along time
    integral_photon = xuv_spectrum.dt * np.sum(xuv_Et_photon.reshape(1,-1) * cross_section_p_sqrt * e_fft, axis=1) # integrate along time
    integral_photon = np.abs(integral_photon)**2
    integral_photon = integral_photon / np.max(integral_photon)


    # integrate along time
    integral_electron = xuv_spectrum.dt * np.sum(xuv_Et_electron.reshape(1,-1) * e_fft, axis=1)
    integral_electron = np.abs(integral_electron)**2
    integral_electron = integral_electron / np.max(integral_electron)

    plt.figure(5, figsize=(5,5))
    plt.plot(np.squeeze(p), integral_electron, "b", label=r"$|\int_{-\infty}^{\infty}E_{XUV}^{Electron}(t)\cdot e^{-i  (\frac{p^2}{2} + I_p) t} dt |^2$")
    plt.plot(np.squeeze(p), integral_photon, "r--", label=r"$|\int_{-\infty}^{\infty}E_{XUV}^{Photon}(t) \cdot \sqrt{\sigma(p)} \cdot e^{-i  (\frac{p^2}{2} + I_p) t} dt |^2$")
    plt.xlabel("momentum $p$ [a.u.]")
    # plt.gca().subplots_adjust(top=0.5)
    plt.gca().text(0.7, 0.8, r"$I_p = {} $[a.u.]".format(Ip), transform=plt.gca().transAxes)
    plt.gca().legend(loc=1, prop={'size':15})
    plt.show()
    plt.gcf().savefig("./3electron_photon_spectrum_Ip{}.png".format(Ip))
    exit()

    # ----with tensorflow----
    # calculate E(t)
    xuv_coefs = tf.placeholder(tf.float32, shape=[None, 5])
    # photon
    xuv_cropped_f_in = tf_functions.xuv_taylor_to_E(xuv_coefs)["f_photon_cropped"][0]
    # electron
    # xuv_cropped_f_in = tf_functions.xuv_taylor_to_E(xuv_coefs)["f_cropped"][0]
    paddings_xuv = tf.constant(
        [[xuv_spectrum.indexmin, xuv_spectrum.N - xuv_spectrum.indexmax]], dtype=tf.int32)
    padded_xuv_f = tf.pad(xuv_cropped_f_in, paddings_xuv)

    xuv_time_domain = tf_ifft(tensor=padded_xuv_f, shift=int(xuv_spectrum.N / 2))
    # xuv_time_domain = xuv_time_domain / tf.reduce_max(xuv_time_domain)

    # integrate

    # no dipole
    # product = 1 * tf.reshape(xuv_time_domain, [1,-1]) * 1 * e_fft
    # dipole
    product = 1 * tf.reshape(xuv_time_domain, [1,-1]) * 1 * cross_section_p_sqrt * e_fft

    integral_tf = tf.constant(xuv_spectrum.dt, dtype=tf.complex64) * tf.reduce_sum(product, axis=1)
    integral_tf = tf.square(tf.abs(integral_tf))
    integral_tf = integral_tf / tf.reduce_max(integral_tf)
    with tf.Session() as sess:

        feed_dict = { xuv_coefs:np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]) }
        integral_tf_out = sess.run(integral_tf, feed_dict=feed_dict)
        xuv_time_domain_out = sess.run(xuv_time_domain, feed_dict=feed_dict)
        padded_xuv_f_out = sess.run(padded_xuv_f, feed_dict=feed_dict)

    plt.figure(6)
    plt.plot(np.squeeze(p), integral_tf_out)
    # plt.gcf().savefig("./456_electron_no_dipole")
    plt.gcf().savefig("./456_photon_with_dipole")

    plt.show()
