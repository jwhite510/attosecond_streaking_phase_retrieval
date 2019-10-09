import csv
import pickle
import matplotlib.pyplot as plt
import scipy.constants as sc
import numpy as np
import scipy.interpolate
import os
import sys
import phase_parameters.params as phase_params


def open_data_file(filepath):
    x = []
    y = []
    with open(filepath, 'r') as file:
        for line in file.readlines():
            values = line.rstrip().split(",")
            values = [float(e) for e in values]
            x.append(values[0])
            y.append(values[1])
    return x, y



def interp_measured_data_to_linear(electronvolts_in, intensity_in, plotting=False):
    # convert eV to joules
    joules = np.array(electronvolts_in) * sc.electron_volt  # joules
    hertz = np.array(joules / sc.h)
    Intensity = np.array(intensity_in)

    # define tmat and fmat
    # N = 1024
    N = int(2 * 1024)
    tmax = 1600e-18
    # tmax = 800e-18
    dt = 2 * tmax / N
    tmat = dt * np.arange(-N / 2, N / 2, 1)
    df = 1 / (N * dt)
    fmat = df * np.arange(-N / 2, N / 2, 1)

    # get rid of any negative points if there are any
    Intensity[Intensity < 0] = 0

    # set the edges to 0
    Intensity[-1] = 0
    Intensity[0] = 0

    # for plotting later, reference the values which contain the non-zero Intensity
    f_index_min = hertz[0]
    f_index_max = hertz[-1]

    # add zeros at the edges to match the fmat matrix
    hertz = np.insert(hertz, 0, np.min(fmat))
    Intensity = np.insert(Intensity, 0, 0)
    hertz = np.append(hertz, np.max(fmat))
    Intensity = np.append(Intensity, 0)

    # get the carrier frequency
    f0 = hertz[np.argmax(Intensity)]
    # square root the intensity to get electric field amplitude
    Ef = np.sqrt(Intensity)

    # map the spectrum onto linear fmat
    interpolator = scipy.interpolate.interp1d(hertz, Ef, kind='linear')
    Ef_interp = interpolator(fmat)

    # calculate signal in time
    linear_E_t = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ef_interp)))

    # set the indexes for cropped input
    indexmin = np.argmin(np.abs(fmat - f_index_min))
    indexmax = np.argmin(np.abs(fmat - f_index_max))

    if plotting:
        plt.figure(1)
        plt.plot(hertz, Intensity, color='red')
        plt.plot(fmat, np.zeros_like(fmat), color='blue')

        plt.figure(2)
        plt.plot(fmat, Ef_interp, label='|E|')
        plt.plot(hertz, Intensity, label='I')
        plt.xlabel('frequency [Hz]')
        plt.legend(loc=1)

        plt.figure(3)
        plt.plot(tmat, np.real(linear_E_t))

        plt.figure(4)
        plt.plot(fmat, Ef_interp, color='red', alpha=0.5)
        plt.plot(fmat[indexmin:indexmax], Ef_interp[indexmin:indexmax], color='red')
        plt.show()

    output = {}
    output["hertz"] = hertz
    output["linear_E_t"] = linear_E_t
    output["tmat"] = tmat
    output["fmat"] = fmat
    output["Ef_interp"] = Ef_interp
    output["indexmin"] = indexmin
    output["indexmax"] = indexmax
    output["f0"] = f0
    output["N"] = N
    output["dt"] = dt
    return output
    # fmat [hz]
    # return hertz, linear_E_t, tmat, fmat, Ef_interp, indexmin, indexmax, f0, N, dt

def retrieve_spectrum2(plotting=False):
    # open the file
    with open(os.path.dirname(__file__)+'/sample2/Sheet1.csv', 'r') as file:
        reader = csv.reader(file)
        matrix = np.array(list(reader))
        electronvolts = matrix[1:501, 0].astype('float')
        Intensity = matrix[1:501, 1].astype('float')

    # convert eV to joules
    joules = np.array(electronvolts) * sc.electron_volt  # joules
    hertz = np.array(joules / sc.h)
    Intensity = np.array(Intensity)

    # define tmat and famt
    #N = 2*1024
    # N = 1024
    N = int(2 * 1024)
    tmax = 1600e-18
    # tmax = 800e-18
    dt = 2 * tmax / N
    tmat = dt * np.arange(-N / 2, N / 2, 1)
    df = 1 / (N * dt)
    fmat = df * np.arange(-N / 2, N / 2, 1)

    # pad the retrieved values with zeros to interpolate later
    hertz = np.insert(hertz, 0, -6e18)
    Intensity = np.insert(Intensity, 0, 0)
    hertz = np.insert(hertz, -1, 6e18)
    Intensity = np.insert(Intensity, -1, 0)
    Intensity[Intensity < 0] = 0

    # get the carrier frequency
    f0 = hertz[np.argmax(Intensity)]

    # square root the intensity to get electric field amplitude
    Ef = np.sqrt(Intensity)

    # map the spectrum onto 512 points linearly
    interpolator = scipy.interpolate.interp1d(hertz, Ef, kind='linear')
    Ef_interp = interpolator(fmat)

    # calculate signal in time
    linear_E_t = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ef_interp)))

    # set the indexes for cropped input
    indexmin = np.argmin(np.abs(fmat - 1.75e16))
    indexmax = np.argmin(np.abs(fmat - 9.5625e16))



    if plotting:
        plt.figure(1)
        plt.plot(hertz, Intensity, color='red')
        plt.plot(fmat, np.zeros_like(fmat), color='blue')

        plt.figure(2)
        plt.plot(fmat, Ef_interp, label='|E|')
        plt.plot(hertz, Intensity, label='I')
        plt.xlabel('frequency [Hz]')
        plt.legend(loc=1)

        plt.figure(3)
        plt.plot(tmat, np.real(linear_E_t))

        plt.figure(4)
        plt.plot(fmat, Ef_interp, color='red', alpha=0.5)
        plt.plot(fmat[indexmin:indexmax], Ef_interp[indexmin:indexmax], color='red')
        plt.show()



    # convert the xuv params to atomic units
    params = {}
    params['tmat'] = tmat/sc.physical_constants['atomic unit of time'][0]
    params['fmat'] = fmat*sc.physical_constants['atomic unit of time'][0]
    params['Ef'] = Ef_interp
    params['indexmin'] = indexmin
    params['indexmax'] = indexmax
    params['f0'] = f0*sc.physical_constants['atomic unit of time'][0] + 0.2
    params['N'] = N
    params['dt'] = dt/sc.physical_constants['atomic unit of time'][0]

    return params

def retrieve_spectrum3(plotting=False):
    with open(os.path.dirname(__file__)+"/sample3/jie_data/spec.p", "rb") as file:
        spec_data = pickle.load(file)

    # add the Ip to electron eV axis
    spec_data["electron"]["eV"] = np.array(spec_data["electron"]["eV"]) + phase_params.Ip_eV

    electronvolts = spec_data["electron"]["eV"]
    Intensity = spec_data["electron"]["I"]

    hertz, linear_E_t, tmat, fmat, Ef_interp, indexmin, indexmax, f0, N, dt = my_interp(electronvolts_in=electronvolts, intensity_in=Intensity, plotting=plotting)

    electronvolts = spec_data["photon"]["eV"]
    Intensity = spec_data["photon"]["I"]

    _, _, _, _, Ef_interp_photon, _, _, _, _, _= my_interp(electronvolts_in=electronvolts, intensity_in=Intensity, plotting=plotting)




    # convert the xuv params to atomic units
    params = {}
    params['tmat'] = tmat/sc.physical_constants['atomic unit of time'][0]
    params['fmat'] = fmat*sc.physical_constants['atomic unit of time'][0]
    params['Ef'] = Ef_interp
    params['Ef_photon'] = Ef_interp_photon
    params['indexmin'] = indexmin
    params['indexmax'] = indexmax
    params['f0'] = f0*sc.physical_constants['atomic unit of time'][0] + 0.2
    params['N'] = N
    params['dt'] = dt/sc.physical_constants['atomic unit of time'][0]

    return params

def retrieve_spectrum4(plotting=False):

    electron_volts, intensity = open_data_file(os.path.dirname(__file__)+'/sample4/spectrum4_electron.csv')
    # add the ionization potential to the electron volts
    electron_volts = [e+phase_params.Ip_eV for e in electron_volts]

    # normalize intensity
    intensity = np.array(intensity)
    intensity = intensity / np.max(intensity)

    electron_interp = interp_measured_data_to_linear(electronvolts_in=electron_volts, intensity_in=intensity, plotting=plotting)

    # open the cross section
    electron_volts_cs, cross_section = open_data_file(os.path.dirname(__file__)+'/sample4/HeliumCrossSection.csv')

    # interpolate the cross section to match the electron spectrum
    interpolator = scipy.interpolate.interp1d(electron_volts_cs, cross_section, kind='linear')
    cross_sec_interp = interpolator(electron_volts)

    # calculate the photon spectrum by diving by the cross section
    photon_spec_I = intensity / cross_sec_interp

    # normalize photon spec intensity
    photon_spec_I = photon_spec_I / np.max(photon_spec_I)

    # interpolate the photon spectrum
    photon_interp = interp_measured_data_to_linear(electronvolts_in=electron_volts, intensity_in=photon_spec_I, plotting=plotting)

    # convert the xuv params to atomic units
    params = {}
    params['tmat'] = electron_interp["tmat"]/sc.physical_constants['atomic unit of time'][0]
    params['fmat'] = electron_interp["fmat"]*sc.physical_constants['atomic unit of time'][0] # 1 / time [a.u.]
    params['Ef'] = electron_interp["Ef_interp"]
    params['Ef_photon'] = photon_interp["Ef_interp"]
    params['indexmin'] = electron_interp["indexmin"]
    params['indexmax'] = electron_interp["indexmax"]
    params['f0'] = f0*sc.physical_constants['atomic unit of time'][0] + 0.2
    params['N'] = N
    params['dt'] = dt/sc.physical_constants['atomic unit of time'][0]

    return params




#==============================
#========select sample=========
#==============================

spectrum = 4

if spectrum == 2:

    params = retrieve_spectrum2()
    tmat = params['tmat']
    tmat_as = params['tmat'] * sc.physical_constants['atomic unit of time'][0] * 1e18 # attoseconds
    fmat = params['fmat']
    fmat_hz = params['fmat'] / sc.physical_constants['atomic unit of time'][0] # hz
    Ef = params['Ef']
    indexmin = params['indexmin']
    indexmax = params['indexmax']
    f0 = params['f0']
    N = params['N']
    dt = params['dt']
    fmat_cropped = fmat[indexmin: indexmax]
    fmat_hz_cropped = fmat_hz[indexmin: indexmax]

elif spectrum == 3:

    params = retrieve_spectrum3()
    tmat = params['tmat']
    tmat_as = params['tmat'] * sc.physical_constants['atomic unit of time'][0] * 1e18 # attoseconds
    fmat = params['fmat']
    fmat_hz = params['fmat'] / sc.physical_constants['atomic unit of time'][0] # hz
    Ef = params['Ef']
    Ef_photon = params['Ef_photon']
    indexmin = params['indexmin']
    indexmax = params['indexmax']
    f0 = params['f0']
    N = params['N']
    dt = params['dt']
    fmat_cropped = fmat[indexmin: indexmax]
    fmat_hz_cropped = fmat_hz[indexmin: indexmax]

elif spectrum == 4:

    params = retrieve_spectrum4()
    tmat = params['tmat']
    tmat_as = params['tmat'] * sc.physical_constants['atomic unit of time'][0] * 1e18 # attoseconds
    fmat = params['fmat']
    fmat_hz = params['fmat'] / sc.physical_constants['atomic unit of time'][0] # hz
    Ef = params['Ef']
    Ef_photon = params['Ef_photon']
    indexmin = params['indexmin']
    indexmax = params['indexmax']
    f0 = params['f0']
    N = params['N']
    dt = params['dt']
    fmat_cropped = fmat[indexmin: indexmax]
    fmat_hz_cropped = fmat_hz[indexmin: indexmax]

if __name__ == "__main__":

    # spectrum 2
    # params = retrieve_spectrum2(plotting=True)

    # spectrum 3
    params = retrieve_spectrum3(plotting=True)










