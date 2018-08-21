import numpy as np
import matplotlib.pyplot as plt

N = 100
dt = 1
t = dt * np.arange(-N/2, N/2, 1)
df = 1 / (dt * N)
f = df * np.arange(-N/2, N/2, 1)

tau = 5
E = np.exp(-t**2/tau**2) + 5*np.exp(-(t - 10)**2/tau**2)
E_w = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E)))

# defien lin phasse
lin_phase = 100 * f


E_w_p = E_w * np.exp(1j * lin_phase)

E_t_p = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(E_w_p)))



fig, ax = plt.subplots(4, 1)

#1
ax[0].plot(t, E)

#2
ax[1].plot(f, np.real(E_w), color='blue')
ax[1].plot(f, np.imag(E_w), color='red')
axtwin = ax[1].twinx()
axtwin.plot(f, lin_phase, color='green', linestyle='dashed')

#3
ax[2].plot(f, np.real(E_w_p), color='blue')
ax[2].plot(f, np.imag(E_w_p), color='red')

#4
ax[3].plot(t, np.real(E_t_p), color='blue')
ax[3].plot(t, np.imag(E_t_p), color='red')


plt.show()
