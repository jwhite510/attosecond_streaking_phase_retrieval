from crab_tf2 import *
import tables
import csv
from scipy import interpolate



def process_trace(trace, time, p_vec, f0=None, fields=None):

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


    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(5, 2)

    ax_trace = fig.add_subplot(gs[1, :])
    if fields:
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(xuv.tmat, np.real(fields['xuv_t']), color='blue')
        ax.set_title('XUV pulse')
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(ir.tmat, np.real(fields['ir_t']), color='blue')
        ax.set_title('IR pulse')
    else:
        ax_trace.set_title('Measured Trace')

    ax_trace.pcolormesh(time, p_vec, trace, cmap='jet')
    ax_trace.set_ylabel('momentum\np [atomic units]')

    ax = fig.add_subplot(gs[2, :])
    ax.pcolormesh(freq, p_vec, np.abs(trace_ft), cmap='jet')
    ax.text(0,0.9, '|Fourier transform trace|', transform=ax.transAxes, backgroundcolor='yellow')
    ax.set_ylabel('momentum\n$p [atomic units]$')

    ax = fig.add_subplot(gs[3, :])
    ax.plot(freq, integral_p, color='black', label=r'$\int trace(p, f) dp$')
    ax.text(0.8, 0.7, 'find $\omega_L$ by determining\n 2nd and 3rd maximum', transform=ax.transAxes, backgroundcolor='yellow')

    # plot the actual f0, if given
    if f0:
        ax.plot([-f0, -f0], [np.min(integral_p), np.max(integral_p)], color='red', alpha=0.5, linestyle='dashed')
        ax.plot([f0, f0], [np.min(integral_p), np.max(integral_p)], color='red', label='actual f0', alpha=0.5, linestyle='dashed')
        ax.legend(loc=1)

    index = max_ind[1]
    ax.plot([freq[index], freq[index]], [np.min(integral_p), np.max(integral_p)], color='blue')
    index = max_ind[2]
    ax.plot([freq[index], freq[index]], [np.min(integral_p), np.max(integral_p)], color='blue', label='+/- $\omega_L$')
    ax.set_xlim(freq[0], freq[-1])
    ax.legend(loc=4)


    ax = fig.add_subplot(gs[4, 0])

    ax.plot(p_vec, np.real(vecneg), color='blue')
    ax.plot(p_vec, np.imag(vecneg), color='red')
    ax.text(0.3, -0.5, 'plot $\omega_L$ components', transform=ax.transAxes, backgroundcolor='yellow')
    ax.set_xlabel('momentum p [atomic units]')

    ax = fig.add_subplot(gs[4, 1])
    ax.plot(p_vec, np.real(vecpos), color='blue')
    ax.plot(p_vec, np.imag(vecpos), color='red')
    ax.text(0.3, -0.5, 'plot $\omega_L$ components', transform=ax.transAxes, backgroundcolor='yellow')
    ax.set_xlabel('momentum p [atomic units]')


def get_measured_trace():
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


def get_generated_specific_trace(index):

    filename = '../attstrac_specific.hdf5'

    with tables.open_file(filename, mode='r') as hdf5file:

        xuv_f = hdf5file.root.xuv_f[index, :]
        xuv_t = hdf5file.root.xuv_t[index, :]

        ir_t = hdf5file.root.ir_t[index, :]
        ir_f = hdf5file.root.ir_f[index, :]

        fields = {'xuv_f': xuv_f,
                  'xuv_t': xuv_t,
                  'ir_t': ir_t,
                  'ir_f': ir_f,
                  }

        trace = hdf5file.root.trace[index, :].reshape(len(p_values), len(tau_values))


    return tau_values, p_values, trace, fields


def get_process_generated(index):

    tau, p, trace, fields = get_generated_specific_trace(index)
    process_trace(trace, tau, p, fields=fields)
    plt.savefig('./{}.png'.format(index))






# delay_meas, p_vec_meas, trace_meas = get_measured_trace()
# process_trace(trace_meas, delay_meas, p_vec_meas)

# tau, p, trace, fields = get_generated_specific_trace(index=3)
# process_trace(trace, tau, p, fields=fields)


# get_process_generated(3)
# get_process_generated(4)
# get_process_generated(5)
# get_process_generated(6)
# get_process_generated(7)





plt.show()







