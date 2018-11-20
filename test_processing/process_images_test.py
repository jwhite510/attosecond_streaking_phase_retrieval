import numpy as np
import matplotlib.pyplot as plt
import tables
import pickle
import csv
from scipy import interpolate
import scipy.constants as sc


def process_trace(trace, time, p_vec, f0=None):

    N = len(time)
    dt = time[1] - time[0]
    df = 1 / (dt * N)
    freq = df * np.arange(-N/2, N/2, 1)

    # fourier transform the trace
    trace_ft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(trace, axes=1), axis=1), axes=1)

    # find w1
    integral_p = np.sum(np.abs(trace_ft), axis=0)


    # find the second maxima
    integral_p_2 = np.array(integral_p)
    # find the three max values
    max_vals = []
    max_ind = []
    # find the top 3 points
    for _ in range(3):
        current_max_index = np.argmax(integral_p_2)
        current_max = integral_p_2[current_max_index]
        max_vals.append(current_max)
        max_ind.append(current_max_index)
        integral_p_2[current_max_index] = 0

    # use only these frequencies to plot the image
    trace_ft_filtered = np.zeros_like(trace_ft)
    trace_ft_filtered[:, max_ind[1]] = trace_ft[:, max_ind[1]]
    trace_ft_filtered[:, max_ind[2]] = trace_ft[:, max_ind[2]]

    # IR w_l vectors
    if max_ind[1] > max_ind[2]:

        pos_index = max_ind[1]
        neg_index = max_ind[2]

    else:

        pos_index = max_ind[2]
        neg_index = max_ind[1]

    vecneg =  trace_ft[:, neg_index]
    vecpos =  trace_ft[:, pos_index]


    fig = plt.figure()
    gs = fig.add_gridspec(5, 2)

    ax = fig.add_subplot(gs[0, :])
    ax.pcolormesh(time, p_vec, trace, cmap='jet')

    ax = fig.add_subplot(gs[1, :])
    ax.pcolormesh(freq, p_vec, np.abs(trace_ft), cmap='jet')
    ax.text(0,0.9, '|trace ft|', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[2, :])
    ax.plot(freq, integral_p)

    # plot the actual f0, if given
    if f0:
        ax.plot([-f0, -f0], [np.min(integral_p), np.max(integral_p)], color='red', alpha=0.5, linestyle='dashed')
        ax.plot([f0, f0], [np.min(integral_p), np.max(integral_p)], color='red', label='actual f0', alpha=0.5, linestyle='dashed')
        ax.legend(loc=1)

    index = max_ind[1]
    ax.plot([freq[index], freq[index]], [np.min(integral_p), np.max(integral_p)], color='blue')
    index = max_ind[2]
    ax.plot([freq[index], freq[index]], [np.min(integral_p), np.max(integral_p)], color='blue')

    ax = fig.add_subplot(gs[3, 0])
    ax.pcolormesh(freq, p_vec, np.real(trace_ft_filtered), cmap='jet')
    ax.text(0, 0.9, 'real $\omega_1$ trace ft', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[3, 1])
    ax.pcolormesh(freq, p_vec, np.imag(trace_ft_filtered), cmap='jet')
    ax.text(0, 0.9, 'imag $\omega_1$ trace ft', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[4, 0])
    ax.plot(p_vec, np.real(vecneg), color='blue')
    ax.plot(p_vec, np.imag(vecneg), color='red')

    ax = fig.add_subplot(gs[4, 1])
    ax.plot(p_vec, np.real(vecpos), color='blue')
    ax.plot(p_vec, np.imag(vecpos), color='red')








def get_measured_trace(plotting=False):
    filepath = '../experimental_data/53asstreakingdata.csv'
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        matrix = np.array(list(reader))

        Energy = matrix[4:, 0].astype('float')
        Delay = matrix[2, 2:].astype('float')
        Values = matrix[4:, 2:].astype('float')


    # map the function onto a grid matching the training data
    interp2 = interpolate.interp2d(Delay, Energy, Values, kind='linear')
    timespan = np.abs(Delay[-1]) + np.abs(Delay[0])
    delay_new = np.arange(Delay[0], Delay[-1], timespan/176)
    energy_new = np.linspace(Energy[0], Energy[-1], 200)
    values_new = interp2(delay_new, energy_new)

    # interpolate to momentum [a.u]
    energy_new_joules = energy_new * sc.electron_volt # joules
    energy_new_au = energy_new_joules / sc.physical_constants['atomic unit of energy'][0]  # a.u.
    momentum_new_au = np.sqrt(2 * energy_new_au)
    interp2_momentum = interpolate.interp2d(delay_new, momentum_new_au, values_new, kind='linear')

    # interpolate onto linear momentum axis
    N = len(momentum_new_au)
    momentum_linear = np.linspace(momentum_new_au[0], momentum_new_au[-1], N)
    values_lin_momentum = interp2_momentum(delay_new, momentum_linear)


    return delay_new, momentum_linear, values_lin_momentum


delay_meas, p_vec_meas, trace_meas = get_measured_trace()



filename = '../attstrac_specific.hdf5'


with open('../crab_tf_items.p', 'rb') as file:
    crab_tf_items = pickle.load(file)
    p_vec = crab_tf_items['p_vec']
    tauvec = crab_tf_items['tauvec']
    tauvec_time = tauvec * crab_tf_items['dt']
    f0_ir = crab_tf_items['irf0']


# use generate specific data
index = 0
hdf5_file = tables.open_file(filename, mode='r')
xuv = hdf5_file.root.xuv_real[index, :] + 1j * hdf5_file.root.xuv_imag[index, :]
trace = hdf5_file.root.trace[index, :].reshape(len(p_vec), len(tauvec))
hdf5_file.close()


# plot the xuv
_, ax = plt.subplots(2, 1)

ax[0].pcolormesh(trace, cmap='jet')
ax[0].text(0.2, 0.9, 'index: {}'.format(str(index)), transform=ax[0].transAxes, backgroundcolor='white')
ax[1].plot(np.abs(xuv), color='black', linestyle='dashed', alpha=0.5)
ax[1].plot(np.real(xuv), color='blue')
ax[1].plot(np.imag(xuv), color='red')



# process_trace(trace=trace, time=tauvec_time, p_vec=p_vec, f0=f0_ir)
process_trace(trace=trace, time=tauvec_time, p_vec=p_vec)

process_trace(trace=trace_meas, time=delay_meas, p_vec=p_vec_meas)







plt.show()
