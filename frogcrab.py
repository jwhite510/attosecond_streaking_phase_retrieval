import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz





def plot_location(tau_index, p_index, axis_t, axis_p, number, product, terms, tau, p):

    axis_t[0][number].plot(t, np.real(product[p_index, tau_index, :]), color='blue')
    axis_t[0][number].plot(t, np.imag(product[p_index, tau_index, :]), color='red')
    axis_t[0][number].set_xlabel('time')
    integral_time = dt * np.sum(product[p_index, tau_index, :])
    integral_time = np.abs(integral_time)
    axis_t[0][number].text(0.1, 1.1, 'abs of integral along t: {}'.format(integral_time),
               transform=axis_t[0][number].transAxes, backgroundcolor='white')
    axis_t[0][number].text(0.1, 1.05, 'p_index: {}, p: {}'.format(p_index, p[p_index]),
               transform=axis_t[0][number].transAxes, backgroundcolor='white')
    axis_t[0][number].text(0.1, 1.0, 'tau_index: {}, tau: {}'.format(tau_index, tau[tau_index]),
               transform=axis_t[0][number].transAxes, backgroundcolor='white')

    axis_t[0][number].text(0.1, 0.9, number,
               transform=axis_t[0][number].transAxes, backgroundcolor='white')

    text = axis_p.text(tau[tau_index], p[p_index], number, backgroundcolor='white', alpha=0.5)
    text.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='black'))

    axis_t[1][number].plot(t, terms['E_xuv_delay'][p_index, tau_index, :], label='E_xuv_delay',
                           color='blue', alpha=0.5)

    axis_t[1][number].plot(t, terms['d_m'][p_index, tau_index, :], label='d_m',
                           color='black', alpha=0.5)

    axis_t[1][number].plot(t, terms['exp_phi_p_t'][p_index, tau_index, :], label='exp_phi_p_t',
                           color='green', alpha=0.5)

    axis_t[1][number].plot(t, terms['e_ft'][p_index, tau_index, :], label='e_ft',
                           color='red', alpha=0.5)









def plot_streaking_trace(t, E_irt, E_xuv, I_p, dt):

    # construct necessary variables for calculation
    N = len(t)
    df = 1/(N * dt)
    f = df * np.arange(-N/2, N/2)
    mid_index = int(len(f)/2)
    p = (4 * np.pi * f[mid_index:])**(0.5)
    tau = t[:]

    # construct mesh grid for tau, p, and t
    tau_m, p_m, t_m = np.meshgrid(tau, p, t)

    # construct matrix for IR pulse
    E_irt_m = np.ones_like(t_m) * E_irt.reshape(1, 1, -1)

    # construction of A_t
    A_t = -1*dt*np.cumsum(E_irt_m, 2)
    A_t_reverse_int = dt * np.flip(np.cumsum(np.flip(A_t, 2), 2), 2)
    A_t_reverse_int_square = dt * np.flip(np.cumsum(np.flip(A_t**2, 2), 2), 2)

    # construction of phi(p, t) term
    phi_p_t = p_m * A_t_reverse_int + 0.5 * A_t_reverse_int_square
    exp_phi_p_t = np.exp(-1j * phi_p_t)

    # construction of delayed XUV matrix
    E_padded = np.pad(E_xuv[::-1], (0, len(E_xuv) - 1), mode='constant')
    toeplitz_m = np.tril(toeplitz(E_padded, E_xuv))
    length_toepl, _ = np.shape(toeplitz_m)
    middle_index = int((length_toepl-1)/2)
    _, crop_span, _ = np.shape(tau_m)
    crop_side = int((crop_span)/2)
    toeplitz_cropped = toeplitz_m[middle_index-crop_side:middle_index+crop_side, :]
    p_length, _, _ = np.shape(p_m)
    E_xuv_delay = np.array([toeplitz_cropped, ] * p_length)

    # construction of dipole moment term d[p + A(t)]
    d_m_numerator = p_m + A_t
    d_m_denom = ((p_m + A_t)**2 + 2*I_p)**3
    d_m = d_m_numerator / d_m_denom

    # construction of fourier transform exponential e^(j(p^2/2 + I_p)t)
    e_ft = np.exp(1j * ((p_m**2)/2 + I_p) * t_m)

    # multiply all the terms together
    product = E_xuv_delay * d_m * exp_phi_p_t * e_ft

    # integrate the product
    integrated = dt * np.sum(product, 2)
    S = np.abs(integrated)**2

    # plot streaking trace
    fig, ax = plt.subplots(2, 4, figsize=(11, 5))
    ax[0][0].set_ylabel('p')
    ax[0][0].set_xlabel('tau')
    ax[0][0].pcolormesh(tau, p, S, cmap='jet')

    # for plotting individual terms
    terms = {'E_xuv_delay': E_xuv_delay, 'd_m': d_m, 'exp_phi_p_t': exp_phi_p_t, 'e_ft': e_ft}

    # plot visualization aids showing product along time axis before integration
    plot_location(tau_index=128, p_index=10, axis_t=ax, axis_p=ax[0][0], number=1, product=product, terms=terms, tau=tau, p=p)
    plot_location(tau_index=128, p_index=45, axis_t=ax, axis_p=ax[0][0], number=2, product=product, terms=terms, tau=tau, p=p)
    plot_location(tau_index=128, p_index=80, axis_t=ax, axis_p=ax[0][0], number=3, product=product, terms=terms, tau=tau, p=p)

    # plot A(t)
    ax[1][0].plot(t_m[0, 0, :], A_t[0, 0, :], color='blue')
    text = ax[1][0].text(0.1, 0.9, 'A(t)', transform=ax[1][0].transAxes, color='blue')
    text.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='black'))
    plt.show()




tmax = 30e-15
N = 256
dt = (2 * tmax) / N
t = dt * np.arange(-N/2, N/2, 1)

#define IR pulse
width_ir = 20e-15
w0_ir = 5e14
E_irt = 1e21 * np.real(np.exp(-t**2/width_ir**2)*np.exp(1j * w0_ir * t))

#define xuv pulse
width_xuv = 0.5e-15
w0_xuv = 5e15
E_xuv = 1 * np.real(np.exp(-t**2/width_xuv**2)*np.exp(1j * w0_xuv * t))

fig, ax = plt.subplots(2, 1)
ax[0].plot(t, np.real(E_irt), color='blue')
ax[0].plot(t, np.imag(E_irt), color='red')
ax[0].text(0.5, 1, 'E_irt', transform=ax[0].transAxes, backgroundcolor='white')
ax[1].plot(t, np.real(E_xuv), color='blue')
ax[1].plot(t, np.imag(E_xuv), color='red')
ax[1].text(0.5, 1, 'E_xuv', transform=ax[1].transAxes, backgroundcolor='white')

I_p = 12e15

plot_streaking_trace(t, E_irt, E_xuv, I_p, dt)

plt.show()
