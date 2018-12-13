import csv
import matplotlib.pyplot as plt
import scipy.constants as sc
import numpy as np
import scipy.interpolate
import pickle

with open('xuv_spectrum.csv', 'r') as file:
    reader = csv.reader(file)
    x = []
    y = []
    for row in reader:
        x.append(float(row[0]))
        y.append(float(row[1]))



# define fmat and tmat
N = 512

# convert eV to joules
joules = np.array(x) * sc.electron_volt # joules
hertz = joules / sc.h

hertz = np.array(hertz)
y = np.array(y)


tmax = 400e-18
dt = 2 * tmax / N
tmat = dt * np.arange(-N/2, N/2, 1)
df = 1 / (N * dt)
fmat = df * np.arange(-N/2, N/2, 1)

plt.figure(1)
plt.plot(hertz, y, color='black')

# append points
# print(hertz)
hertz = np.insert(hertz, 0, -6e17)
y = np.insert(y, 0, 0)
hertz = np.insert(hertz, -1, 6e17)
y = np.insert(y, -1, 0)
y[y<0] = 0


plt.plot(hertz, y, color='red')
plt.plot(fmat, np.zeros_like(fmat), color='blue')


# map the spectrum onto 512 points linearly
interpolator = scipy.interpolate.interp1d(hertz, y, kind='linear')
Ef = interpolator(fmat)





plt.figure(2)
plt.plot(fmat, Ef)
# plt.plot(Ef)

linear_E_t = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ef)))

plt.figure(3)
plt.plot(tmat, np.real(linear_E_t))
# plt.plot(np.real(linear_E_t))


pickle_files = {}
pickle_files['tmat'] = tmat
pickle_files['fmat'] = fmat
pickle_files['Ef'] = Ef

# write the data to a pickle
with open('../measured_spectrum.p', 'wb') as file:
    pickle.dump(pickle_files, file)


plt.show()










