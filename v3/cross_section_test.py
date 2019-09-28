import tensorflow as tf
import phase_parameters.params
import scipy.constants as sc
import tf_functions
import numpy as np
import matplotlib.pyplot as plt
import xuv_spectrum.spectrum as xuv_spectrum
import scipy.interpolate


def multiply_by_cross_section():

    # get photon and electron E(f)
    xuv_Ef = xuv_spectrum.Ef
    xuv_Ef_photon = xuv_spectrum.Ef_photon
    fmat_cropped_au_freq = xuv_spectrum.fmat_hz_cropped / sc.physical_constants['atomic unit of time'][0] # au 1/time

    # get cross section
    cross_section = xuv_spectrum.cross_section
    cross_section_au_freq = ev_to_au(xuv_spectrum.cross_section_ev)

    plt.figure(2)
    # plt.title("test")
    plt.plot(fmat_cropped_au_freq, np.abs(xuv_Ef[xuv_spectrum.indexmin:xuv_spectrum.indexmax])**2)
    crop_indexes = (0, -10)
    plt.plot(fmat_cropped_au_freq[crop_indexes[0]:crop_indexes[1]], (np.abs(xuv_Ef[xuv_spectrum.indexmin:xuv_spectrum.indexmax])**2)[crop_indexes[0]:crop_indexes[1]], "r-")
    plt.plot(cross_section_au_freq, cross_section, "g-")

    # interpolate cross section
    cross_section_interpolator = scipy.interpolate.interp1d(cross_section_au_freq, cross_section)
    cross_section_interp = cross_section_interpolator(fmat_cropped_au_freq[crop_indexes[0]:crop_indexes[1]])
    plt.plot(fmat_cropped_au_freq[crop_indexes[0]:crop_indexes[1]], cross_section_interp, "b-")


    # divide the electron spectrum by the photon spectrum
    I_photon = (np.abs(xuv_Ef[xuv_spectrum.indexmin:xuv_spectrum.indexmax])**2)[crop_indexes[0]:crop_indexes[1]] / cross_section_interp
    plt.figure(3)

    # normalize
    I_photon = I_photon / np.max(I_photon)
    plt.plot(fmat_cropped_au_freq[crop_indexes[0]:crop_indexes[1]], I_photon)

    spec_I_photon = (np.abs(xuv_spectrum.Ef_photon[xuv_spectrum.indexmin:xuv_spectrum.indexmax])**2)[crop_indexes[0]:crop_indexes[1]]
    plt.plot(fmat_cropped_au_freq[crop_indexes[0]:crop_indexes[1]], spec_I_photon)

    plt.show()

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

def ev_to_wl(vector):
    """
    ev_to_joules(vector [eV])
    return vector [joules energy]
    """
    vector_joules =  np.array(vector) * sc.electron_volt # joules
    vector_hz = vector_joules / sc.h # Hz
    # c = lambda * f
    wl_lambda = sc.c / vector_hz
    return wl_lambda



def multiply_by_cross_section_integrate():

    # Fourier transform
    # xuv_spectrum.tmat
    # xuv_spectrum.dt
    xuv_Et_electron = np.fft.fftshift(np.fft.fft(np.fft.fftshift(xuv_spectrum.Ef)))
    xuv_Et_photon = np.fft.fftshift(np.fft.fft(np.fft.fftshift(xuv_spectrum.Ef_photon)))


    # momentum in atomic units
    K = phase_parameters.params.K * sc.electron_volt  # joules
    K = K / sc.physical_constants['atomic unit of energy'][0]  # a.u. energy
    p = np.sqrt(2 * K).reshape(-1, 1) # a.u. momentum


    # create interpolator for cross section
    cross_section_au_momentum = ev_to_p(xuv_spectrum.cross_section_ev)
    interpolator = scipy.interpolate.interp1d(cross_section_au_momentum, xuv_spectrum.cross_section, kind='linear')
    cross_section_p = interpolator(np.squeeze(p))
    cross_section_p = cross_section_p / np.max(cross_section_p)

    # integrate
    integral_photon = xuv_spectrum.dt * np.sum(xuv_Et_photon.reshape(1,-1) * np.sqrt(cross_section_p.reshape(-1,1)) * np.exp(1j*((p**2)/2)*xuv_spectrum.tmat), axis=1) # integrate along time
    integral_photon = np.abs(integral_photon)**2
    integral_photon = integral_photon / np.max(integral_photon)

    integral_electron = xuv_spectrum.dt * np.sum(xuv_Et_electron.reshape(1,-1) * np.exp(1j*((p**2)/2)*xuv_spectrum.tmat), axis=1) # integrate along time
    integral_electron = np.abs(integral_electron)**2
    integral_electron = integral_electron / np.max(integral_electron)

    plt.figure(3)
    plt.plot(np.squeeze(p), integral_photon, "b")
    plt.title(r"$| \int_{-\infty}^{\infty} E_{xuv}^{photon}(t) \sqrt{\sigma(p)} e^{i(\frac{p^2}{2})t} dt |^2$")
    plt.xlabel("momentum p [a.u.]")
    plt.gcf().savefig("./photon_integral.png")

    plt.figure(4)
    plt.plot(np.squeeze(p), integral_electron, "b")
    plt.title(r"$| \int_{-\infty}^{\infty} E_{xuv}^{electron}(t) e^{i(\frac{p^2}{2})t} dt |^2$")
    plt.xlabel("momentum p [a.u.]")
    plt.gcf().savefig("./electron_integral.png")

    # plot cross section
    henke_cross_section_ev, henke_cross_section_f2 = open_he_cross_section()
    henke_cross_section_p = ev_to_p(henke_cross_section_ev)
    wl_lambda = ev_to_wl(henke_cross_section_ev)
    r_e = sc.physical_constants['classical electron radius'][0]
    # calcualte cross section

    # henke_cross_section_wl_lambda = # meters
    henke_cross_section_sigma = 2 * r_e * wl_lambda * henke_cross_section_f2
    # interpolate henke cross section onto p values

    henke_interp = scipy.interpolate.interp1d(henke_cross_section_p, henke_cross_section_sigma)
    henk_vals_interp_sigma_p = henke_interp(np.squeeze(p))
    henk_vals_interp_sigma_p = henk_vals_interp_sigma_p / np.max(henk_vals_interp_sigma_p)

    plt.figure(5)
    plt.plot(np.squeeze(p), cross_section_p, "b-",label="cross section\n Jie gave me ")
    plt.plot(np.squeeze(p), henk_vals_interp_sigma_p, "r-", label="cross section\ncalculated from data on henke.lbl.gov\n"+r"$\sigma = (2\cdot r_e\cdot  \lambda \cdot f_2)$")
    plt.ylabel("$\sigma(p)$")
    plt.xlabel("momentum p[a.u.]")
    l = plt.gca().legend()
    l.get_texts()[0].set_color("blue")
    l.get_texts()[1].set_color("red")
    plt.gcf().savefig("./cross_sections.png")


    plt.show()


def open_he_cross_section():

    with open("./xuv_spectrum/sample4/henke_data/he.txt", "r") as file:
        f2_vals = []
        E_eV_vals = []
        for line in list(file.readlines())[1:]:
            f2_val = line.split()[2]
            f2_vals.append(f2_val)
            E_eV_val = line.split()[0]
            E_eV_vals.append(E_eV_val)

        E_eV_vals = [float(e) for e in E_eV_vals]
        f2_vals = [float(e) for e in f2_vals]

        E_eV_vals = np.array(E_eV_vals)
        f2_vals = np.array(f2_vals)

    return E_eV_vals, f2_vals



if __name__ == "__main__":
    multiply_by_cross_section_integrate()

    # multiply_by_cross_section()

    # open_he_cross_section()



