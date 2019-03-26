import csv
import matplotlib.pyplot as plt
import scipy.constants as sc
import numpy as np
import scipy.interpolate
import os


def retrieve_spectrum(plotting=False):
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
    N = 2*1024
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


params = retrieve_spectrum()
tmat = params['tmat']
fmat = params['fmat']
Ef = params['Ef']
indexmin = params['indexmin']
indexmax = params['indexmax']
f0 = params['f0']
N = params['N']
dt = params['dt']
fmat_cropped = fmat[indexmin: indexmax]


if __name__ == "__main__":

    params = retrieve_spectrum(plotting=True)


