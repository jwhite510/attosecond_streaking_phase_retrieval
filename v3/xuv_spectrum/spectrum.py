import csv
import pickle
import matplotlib.pyplot as plt
import scipy.constants as sc
import numpy as np
import scipy.interpolate
import os
import sys
import phase_parameters.params as phase_params


def my_interp(electronvolts_in, intensity_in, plotting=False):
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

    # pad the vectors with zeros
    hertz = np.insert(hertz, 0, hertz[0])
    Intensity = np.insert(Intensity, 0, 0)
    hertz = np.append(hertz, hertz[-1])
    Intensity = np.append(Intensity, 0)

    # pad the retrieved values with zeros to interpolate later
    hertz = np.insert(hertz, 0, -6e18)
    Intensity = np.insert(Intensity, 0, 0)
    hertz = np.append(hertz, 6e18)
    Intensity = np.append(Intensity, 0)
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
    # indexmin = np.argmin(np.abs(fmat - 1.75e16))
    indexmin = np.argmin(np.abs(fmat - 1.26e16))
    # indexmax = np.argmin(np.abs(fmat - 7.99e16))
    indexmax = np.argmin(np.abs(fmat - 9.34e16))

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

    return hertz, linear_E_t, tmat, fmat, Ef_interp, indexmin, indexmax, f0, N, dt

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
    electron_volts = []
    intensity = []
    with open(os.path.dirname(__file__)+'/sample4/spectrum4_electron.csv', 'r') as file:
        for line in file.readlines():
            values = line.rstrip().split(",")
            values = [float(e) for e in values]
            electron_volts.append(values[0])
            intensity.append(values[2])

    # add the ionization potential to the electron volts
    electron_volts = [e+phase_params.Ip_eV for e in electron_volts]

    # normalize intensity
    intensity = np.array(intensity)
    intensity = intensity / np.max(intensity)

    hertz, linear_E_t, tmat, fmat, Ef_interp, indexmin, indexmax, f0, N, dt = my_interp(electronvolts_in=electron_volts, intensity_in=intensity, plotting=plotting)

    # calculate photon spectrum
    electron_volts_cs = []
    cross_section = []
    with open(os.path.dirname(__file__)+'/sample4/HeliumCrossSection.csv', 'r') as file:
        for line in file.readlines():
            values = line.rstrip().split(",")
            values = [float(e) for e in values]
            cross_section.append(values[1])
            electron_volts_cs.append(values[0])

    # append the cross section values because the inteprolation is out of range
    cs_append_value = float(cross_section[-1])
    d_ev = electron_volts_cs[-1] - electron_volts_cs[-2]
    for _ in range(15):
        ev_append = electron_volts_cs[-1] + d_ev
        cross_section.append(cs_append_value)
        electron_volts_cs.append(ev_append)

    # interpolate the cross section to match the electron spectrum
    interpolator = scipy.interpolate.interp1d(electron_volts_cs, cross_section, kind='linear')
    cross_sec_interp = interpolator(electron_volts)

    # calculate the photon spectrum by diving by the cross section
    photon_spec_I = intensity / cross_sec_interp

    # normalize photon spec intensity
    photon_spec_I = photon_spec_I / np.max(photon_spec_I)


    # interpolate the photon spectrum
    _, _, _, _, Ef_interp_photon, _, _, _, _, _= my_interp(electronvolts_in=electron_volts, intensity_in=photon_spec_I, plotting=plotting)

    # plt.figure(10)
    # plt.plot(electron_volts_cs, cross_section)
    # plt.title("cross section")

    # plt.figure(11)
    # plt.plot(electron_volts, intensity)
    # plt.title("intensity")

    # plt.figure(12)
    # plt.plot(electron_volts, cross_sec_interp)
    # plt.title("interpolated cross cross section")

    # plt.figure(13)
    # plt.plot(electron_volts, photon_spec_I)
    # plt.title("photon spectrum")

    # plt.figure(14)
    # plt.plot(fmat, Ef_interp)
    # plt.title("linear photon spectrum")

    # plt.figure(15)
    # plt.plot(fmat, Ef_interp_photon)
    # plt.title("linear photon spectrum")

    # plt.show()
    # exit()

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










