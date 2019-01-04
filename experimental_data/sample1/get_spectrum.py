import csv
import matplotlib.pyplot as plt
import scipy.constants as sc
import numpy as np
import scipy.interpolate
import pickle

with open('xuv_spectrum.csv', 'r') as file:
    reader = csv.reader(file)
    electronvolts = []
    Intensity = []
    for row in reader:
        electronvolts.append(float(row[0]))
        Intensity.append(float(row[1]))



# convert eV to joules
joules = np.array(electronvolts) * sc.electron_volt # joules
hertz = np.array(joules / sc.h)
Intensity = np.array(Intensity)

# define tmat and famt
N = 1024
tmax = 800e-18
dt = 2 * tmax / N
tmat = dt * np.arange(-N/2, N/2, 1)
df = 1 / (N * dt)
fmat = df * np.arange(-N/2, N/2, 1)


# pad the retrieved values with zeros to interpolate later
hertz = np.insert(hertz, 0, -6e18)
Intensity = np.insert(Intensity, 0, 0)
hertz = np.insert(hertz, -1, 6e18)
Intensity = np.insert(Intensity, -1, 0)
Intensity[Intensity<0] = 0

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
span = 405 - 280
indexmin = 540
indexmax = int(indexmin + span)



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



# pickle the files for use in xuv pulse generation
pickle_files = {}
pickle_files['tmat'] = tmat
pickle_files['fmat'] = fmat
pickle_files['Ef'] = Ef_interp
pickle_files['indexmin'] = indexmin
pickle_files['indexmax'] = indexmax
pickle_files['f0'] = f0
pickle_files['N'] = N
pickle_files['dt'] = dt

# write the data to a pickle
with open('../measured_spectrum.p', 'wb') as file:
    pickle.dump(pickle_files, file)


plt.show()










