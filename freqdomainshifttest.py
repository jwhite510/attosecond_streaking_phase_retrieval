import numpy as np
import matplotlib.pyplot as plt



def defineE(N, tmax):
    dt = 2 * tmax / N
    t = dt * np.arange(-N/2, N/2, 1)
    f0 = 5e13
    tau = 20e-15
    df = 1 / (dt * N)
    f = df * np.arange(-N/2, N/2, 1)

    E_t = np.exp(-t**2 / tau**2) * np.exp(1j * 2 * np.pi * f0 * t)


    return E_t, t, f



E_t, t, f = defineE(N=64, tmax=50e-15)

E_t2, t2, f2 = defineE(N=256, tmax=50e-15)

start_index = 0
print(t[0:2])
print(t2[0:5])


dtsmall = t2[1] - t2[0]


# construct delay with E_t, t
E_f = np.fft.fftshift(np.fft.fft(np.fft.fftshift(E_t)))
# plt.figure(1)
# plt.plot(f, np.real(E_f))

# construct delay values
delay = dtsmall * np.arange(-3, 4, 1)
# print('delay:\n', delay)
# construct 2d delay matrix
delaymat = delay.reshape(-1, 1) * f.reshape(1, -1)

delayedEf = E_f.reshape(1, -1) * np.exp(1j * 2 * np.pi * delaymat)
delayedEt = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(delayedEf, axes=1), axis=1), axes=1)
# delayedEt = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(delayedEf)))

print('np.shape(delayedEf):', np.shape(delayedEf))
print('np.shape(delaymat):', np.shape(delaymat))
print('delay:', delay)


# pick a point on both pulses (small and large) to start from
t_index = 30
point = E_t[t_index]
t2_index = np.where(E_t2==point)[0][0]


fig = plt.figure()
gs = fig.add_gridspec(3,2)

# plot the delayed signal
ax = fig.add_subplot(gs[0,0])
ax.pcolormesh(np.real(delayedEt))

# plot the original signal in time
ax = fig.add_subplot(gs[1,0])
ax.plot(t, np.real(delayedEt[3, :]))
# plot the signal with 0 delay (should match the original)
ax.plot(t, np.real(E_t), linestyle='dashed')
ax.plot(t[t_index], np.real(E_t[t_index]), 'ro')

# plot the pulse with small dt
ax = fig.add_subplot(gs[1,1])
ax.plot(t2, np.real(E_t2))
ax.plot(t2[t2_index], np.real(E_t2[t2_index]), 'ro')

# plot the small area around the selected point
ax = fig.add_subplot(gs[2,1])
ax.plot(t2[t2_index-3:t2_index+4], np.real(E_t2[t2_index-3:t2_index+4]))
ax.plot(t2[t2_index], np.real(E_t2[t2_index]), 'ro')

# plot the delay values for the point
ax = fig.add_subplot(gs[2,0])
ax.plot(delay+t[t_index], np.real(delayedEt[:, t_index]), color='orange')
ax.plot(delay[3]+t[t_index], np.real(delayedEt[3, t_index]), 'ro')


plt.show()










