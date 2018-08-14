import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz



def plot_streaking_trace(t, E_irt, E_xuv, I_p, dt):

    # construct frequency space
    N = len(t)
    dp = 1/(N * dt)
    p = dp * np.arange(-N/2, N/2, 1)
    tau = t[:]



    tau_m, p_m, t_m = np.meshgrid(tau, p, t)


    E_irt_m = np.ones_like(t_m) * E_irt.reshape(1, 1, -1)

    # define A(t)
    A_t = -1*dt*np.cumsum(E_irt_m, 2)
    # print('A_t:\n ', A_t, '\n')

    # define phi_p_t
    A_t_reverse_int = dt * np.flip(np.cumsum(np.flip(A_t, 2), 2), 2)
    A_t_reverse_int_square = dt * np.flip(np.cumsum(np.flip(A_t**2, 2), 2), 2)

    phi_p_t = p_m * A_t_reverse_int + 0.5 * A_t_reverse_int_square

    E_padded = np.pad(E_xuv[::-1], (0, len(E_xuv) - 1), mode='constant')

    toeplitz_m = np.tril(toeplitz(E_padded, E_xuv))

    length_toepl, _ = np.shape(toeplitz_m)

    middle_index = int((length_toepl-1)/2)
    _, crop_span, _ = np.shape(tau_m)
    crop_side = int((crop_span-1)/2)

    toeplitz_cropped = toeplitz_m[middle_index-crop_side+0:middle_index+crop_side+2, :]
    toeplitz_cropped_flipped = np.flip(toeplitz_cropped, 0)

    p_length, _, _ = np.shape(p_m)
    # print('toeplitz cropped flipped: \n', toeplitz_cropped_flipped, '\n')
    # print(tau_m)
    E_xuv_delay = np.array([toeplitz_cropped, ] * p_length)

    # construct dipole moment
    d_m_numerator = p_m + A_t
    d_m_denom = ((p_m + A_t)**2 + 2*I_p)**3
    d_m = d_m_numerator / d_m_denom

    e_ft = np.exp(1j * ((p_m**2)/2 + I_p) * t_m)

    product = E_xuv_delay * d_m * np.exp(-1j * phi_p_t) * e_ft
    # print('product shape: \n', np.shape(product), '\n')

    integrated = dt * np.sum(product, 2)
    S = np.abs(integrated)**2
    plt.figure(2)
    plt.pcolormesh(S)
    plt.show()


tmax = 50e-15
N = 128
dt = (2 * tmax) / N
t = dt * np.arange(-N/2, N/2, 1)



#define IR pulse
width_ir = 20e-15
w0_ir = 5e14
E_irt = np.real(np.exp(-t**2/width_ir**2)*np.exp(1j * w0_ir * t))
# E_irt = np.array([1, 2, 3, 4, 5, 6])

#define xuv pulse
width_xuv = 5e-15
w0_xuv = 15e14
E_xuv = np.real(np.exp(-t**2/width_xuv**2)*np.exp(1j * w0_xuv * t))
# E_xuv = np.array([1, 2, 3, 4, 5, 6])


fig, ax = plt.subplots(2, 1)
ax[0].plot(t, E_irt)
ax[0].text(0.5, 1, 'E_irt', transform=ax[0].transAxes, backgroundcolor='white')
ax[1].plot(t, E_xuv)
ax[1].text(0.5, 1, 'E_xuv', transform=ax[1].transAxes, backgroundcolor='white')

# print(np.shape(t))
# print(np.shape(E_irt))
# print(np.shape(E_xuv))

I_p = 1e16

# t = np.array([-4, -3, -2, -1, 0, 1, 2, 3])
# E_irt = 10 * np.array([1, -1, 1, -1, 1, -1, 1, -1])
# E_xuv = np.array([-1, 1, -1, 1, -1, 1, -1, 1])
# dt = 1


plot_streaking_trace(t, E_irt, E_xuv, I_p, dt)

plt.show()
