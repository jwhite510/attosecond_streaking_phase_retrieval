import numpy as np
import matplotlib.pyplot as plt
import tables
import pickle
import os

"""
This python file shows detailed plots of the generation of the attosecond trace and the PROOF trace
"""



def plot_xuv_att_proof():

    # plot only the xuv, attosecond trace, and proof trace
    fig = plt.figure(constrained_layout=False, figsize=(7, 7))
    gs = fig.add_gridspec(4, 2)
    # plot the XUV
    ax = fig.add_subplot(gs[0, :])
    ax.plot(np.abs(xuv), color='black', linestyle='dashed', alpha=0.5)
    ax.plot(np.real(xuv), color='blue')
    ax.text(0, 0.9, 'XUV', transform=ax.transAxes, backgroundcolor='white')

    # plot the trace
    ax = fig.add_subplot(gs[1, :])
    ax.pcolormesh(tauvec_time, p_vec, trace, cmap='jet')
    ax.text(0, 0.9, 'Attosecond Streaking Trace', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[2, :])
    ax.pcolormesh(tauvec_time, p_vec, np.abs(trace_filtered_time), cmap='jet')
    ax.text(0, 0.8, 'abs of trace [Delay Domain]', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[3, :])
    ax.pcolormesh(tauvec_time, p_vec, np.real(trace_filtered_time), cmap='jet')
    ax.text(0, 0.8, 'real of trace [Delay Domain]', transform=ax.transAxes, backgroundcolor='white')

    save_figure(filename='xuv_ir_proof_streak')


def plot_compensation():

    # plot the compensation
    fig = plt.figure(constrained_layout=False, figsize=(7, 7))
    gs = fig.add_gridspec(3, 2)
    ax = fig.add_subplot(gs[0, :])
    ax.pcolormesh(tauvec_time, p_vec, np.abs(trace_filtered_time), cmap='jet')
    ax.text(0, 0.9, 'abs of trace', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[1, :])
    ax.pcolormesh(tauvec_time, p_vec, xuv_spectrum_compensation, cmap='jet')
    ax.text(0, 0.9, 'Apply compensation for XUV spectrum', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[2, :])
    ax.pcolormesh(tauvec_time, p_vec, np.abs(trace_compensated), cmap='jet')
    ax.text(0, 0.9, 'trace with compensation', transform=ax.transAxes, backgroundcolor='white')

    save_figure(filename='compensation')


def construct_proof(trace, tauvec, dt, f0_ir):

    # define parameters needed for calculation
    tauvec_time = tauvec * dt
    dtau = tauvec_time[1] - tauvec_time[0]
    Ntau = len(tauvec_time)
    df_tau = 1 / (Ntau * dtau)  # frequency in au
    tauvec_f_space = df_tau * np.arange(-Ntau / 2, Ntau / 2, 1)

    # fourier transform the trace
    traceft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(trace, axes=1), axis=1), axes=1)

    # construct filter
    filter_type = 'rect'

    if filter_type == 'rect':
        width = 2
        filter = np.zeros_like(tauvec_f_space)
        f_pos_index = np.argmin(np.abs(tauvec_f_space - f0_ir))
        f_neg_index = np.argmin(np.abs(tauvec_f_space + f0_ir))
        filter[f_pos_index - width:f_pos_index + width] = 1
        filter[f_neg_index - width:f_neg_index + width] = 1

    elif filter_type == 'gaussian':
        width = 0.001
        filter = np.exp(-(tauvec_f_space - f0_ir) ** 2 / width ** 2)
        filter += np.exp(-(tauvec_f_space + f0_ir) ** 2 / width ** 2)

    filter2d = filter.reshape(1, -1) * np.real(np.ones_like(traceft))
    trace_filtered = filter2d * traceft

    # fourier transform the filtered signal
    trace_filtered_time = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(trace_filtered, axes=1), axis=1), axes=1)

    params = {}
    params['tauvec_f_space'] = tauvec_f_space
    params['filter'] = filter
    params['filter2d'] = filter2d
    params['traceft'] = traceft
    params['trace_filtered'] = trace_filtered
    params['tauvec_time'] = tauvec_time

    return trace_filtered_time, params



def plot_filtering():

    # plot the filtering
    fig = plt.figure(constrained_layout=False, figsize=(7, 7))
    gs = fig.add_gridspec(5, 2)

    # plot the XUV
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(xuv_int_t, np.real(xuv), color='blue')
    ax.text(0, 0.9, 'XUV', transform=ax.transAxes, backgroundcolor='white')

    # plot the trace
    ax = fig.add_subplot(gs[0, 1])
    ax.pcolormesh(tauvec_time, p_vec, trace, cmap='jet')
    ax.text(0, 0.9, 'Attosecond Streaking Trace', transform=ax.transAxes, backgroundcolor='white')

    # plot the fourier transform of the trace
    ax = fig.add_subplot(gs[1, :])
    ax.pcolormesh(tauvec_f_space, p_vec, np.abs(traceft), cmap='jet')
    ax.text(0, 0.9, 'Fourier Transform of Trace', transform=ax.transAxes, backgroundcolor='white')

    # plot the filter
    ax = fig.add_subplot(gs[2, :])
    ax.pcolormesh(tauvec_f_space, p_vec, filter2d, cmap='jet')
    ax.text(0, 0.8, 'Apply filter', transform=ax.transAxes, backgroundcolor='white')
    ax.text(0.58, 0.5, '$+\omega_1$', transform=ax.transAxes, backgroundcolor='white')
    ax.text(0.36, 0.5, '$-\omega_1$', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[3, :])
    ax.pcolormesh(tauvec_f_space, p_vec, np.abs(trace_filtered), cmap='jet')
    ax.text(0, 0.8, 'abs of trace [Frequency Domain]', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[4, :])
    ax.pcolormesh(tauvec_time, p_vec, np.abs(trace_filtered_time), cmap='jet')
    ax.text(0, 0.8, 'abs of trace [Delay Domain]', transform=ax.transAxes, backgroundcolor='white')
    save_figure(filename='filtering')



def plot_ir_xuv_atto():

    # plot the IR, xuv, and attosecond
    fig = plt.figure(constrained_layout=False, figsize=(7, 7))
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(irtmat, irEt, color='orange')
    ax.text(0, 0.9, 'IR pulse', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(xuv_int_t, np.real(xuv), color='blue')
    ax.text(0, 0.9, 'XUV pulse', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[1, :])
    ax.pcolormesh(tauvec_time, p_vec, trace, cmap='jet')
    ax.text(0, 0.9, 'Attosecond Streaking Trace', transform=ax.transAxes, backgroundcolor='white')

    save_figure(filename='xuv_ir_streak')




def plot_before_after_compensation():

    # plot the trace before and after copmensation
    tau_cross_section = 94
    fig = plt.figure(constrained_layout=False, figsize=(10, 7))
    gs = fig.add_gridspec(5, 2)

    ax = fig.add_subplot(gs[0, :])
    ax.pcolormesh(tauvec_time, p_vec, np.abs(trace_filtered_time), cmap='jet')
    ax.text(0, 0.8, 'abs of trace', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[1, 0])
    vals = np.abs(trace_filtered_time)
    p_slice = np.array(vals[:, tau_cross_section])
    vals[:, tau_cross_section] = np.max(vals)
    ax.pcolormesh(tauvec_time, p_vec, vals, cmap='jet')
    ax.text(0, 0.8, 'trace with compensation', transform=ax.transAxes, backgroundcolor='white')
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(p_vec, p_slice)

    ax = fig.add_subplot(gs[2, 0])
    ax.pcolormesh(tauvec_time, p_vec, xuv_spectrum_compensation, cmap='jet')
    ax.text(0, 0.8, 'xuv spectral compensation', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(p_vec, compensation)
    ax.text(0, -0.1, 'xuv spectral compensation offset:{}'.format(offset), transform=ax.transAxes,
            backgroundcolor='white')

    ax = fig.add_subplot(gs[3, :])
    ax.pcolormesh(tauvec_time, p_vec, np.abs(trace_compensated), cmap='jet')
    ax.text(0, 0.8, 'trace with compensation', transform=ax.transAxes, backgroundcolor='white')

    # plot a cross section of the values
    ax = fig.add_subplot(gs[4, 0])
    vals = np.abs(trace_compensated)
    p_slice = np.array(vals[:, tau_cross_section])
    vals[:, tau_cross_section] = np.max(vals)
    ax.pcolormesh(tauvec_time, p_vec, vals, cmap='jet')
    ax.text(0, 0.8, 'trace with compensation', transform=ax.transAxes, backgroundcolor='white')
    ax = fig.add_subplot(gs[4, 1])
    ax.plot(p_vec, p_slice)

    save_figure('detail_compensate')




def plot_before_after_filter():

    # plot the trace before and after filtering
    fig = plt.figure(constrained_layout=False, figsize=(10, 10))
    gs = fig.add_gridspec(5, 3)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(np.abs(xuv), color='black', linestyle='dashed', alpha=0.5)
    ax.plot(np.real(xuv), color='blue')
    ax.plot(np.imag(xuv), color='red')

    ax = fig.add_subplot(gs[0, 1])
    ax.pcolormesh(trace, cmap='jet')
    ax.text(0.2, 0.9, 'index: {}'.format(str(index)), transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[1, 0])
    ax.pcolormesh(tauvec_f_space, p_vec, np.real(traceft), cmap='jet')
    ax.text(0, 0.8, 'real part of trace', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[1, 1])
    ax.pcolormesh(tauvec_f_space, p_vec, np.imag(traceft), cmap='jet')
    ax.text(0, 0.8, 'imag part of trace', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[1, 2])
    ax.pcolormesh(tauvec_f_space, p_vec, np.abs(traceft), cmap='jet')
    ax.text(0, 0.8, 'abs of trace', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[2, 1])
    ax.pcolormesh(tauvec_f_space, p_vec, filter2d, cmap='jet')
    ax.text(0, 0.8, 'filter', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[2, 0])
    plt.plot(tauvec_f_space, filter, color='blue')
    ax.text(0, 0.8, 'filter', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[3, 0])
    ax.pcolormesh(tauvec_f_space, p_vec, np.real(trace_filtered), cmap='jet')
    ax.text(0, 0.8, 'real part of trace', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[3, 1])
    ax.pcolormesh(tauvec_f_space, p_vec, np.imag(trace_filtered), cmap='jet')
    ax.text(0, 0.8, 'imag part of trace', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[3, 2])
    ax.pcolormesh(tauvec_f_space, p_vec, np.abs(trace_filtered), cmap='jet')
    ax.text(0, 0.8, 'abs of trace', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[4, 0])
    ax.pcolormesh(tauvec_time, p_vec, np.real(trace_filtered_time), cmap='jet')
    ax.text(0, 0.8, 'real part of trace', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[4, 1])
    ax.pcolormesh(tauvec_time, p_vec, np.imag(trace_filtered_time), cmap='jet')
    ax.text(0, 0.8, 'imag part of trace', transform=ax.transAxes, backgroundcolor='white')

    ax = fig.add_subplot(gs[4, 2])
    ax.pcolormesh(tauvec_time, p_vec, np.abs(trace_filtered_time), cmap='jet')
    ax.text(0, 0.8, 'abs of trace', transform=ax.transAxes, backgroundcolor='white')

    save_figure('detail_proof')



def save_figure(filename):
    if not os.path.isdir('./'+filename+'/'):
        print('creating '+filename+' folder')
        os.makedirs('./'+filename+'')
    plt.subplots_adjust(left=0.05, right=0.95)

    plt.savefig('./'+filename+'/'+filename+'{}.png'.format(index))


# open the pickles
try:
    with open('crab_tf_items.p', 'rb') as file:
        crab_tf_items = pickle.load(file)

    items = crab_tf_items['items']
    xuv_int_t = crab_tf_items['xuv_int_t']
    tmax = crab_tf_items['tmax']
    N = crab_tf_items['N']
    dt = crab_tf_items['dt']
    tauvec = crab_tf_items['tauvec']
    p_vec = crab_tf_items['p_vec']
    f0_ir = crab_tf_items['irf0']
    irEt = crab_tf_items['irEt']
    irtmat = crab_tf_items['irtmat']

except Exception as e:
    print(e)
    print('run crab_tf.py first to pickle the needed files')
    exit(0)


# open the file and retrieve the trace

if __name__ == '__main__':

    home = os.getcwd()

    for index in np.arange(5, 15, 1):
        print('plotting index {}'.format(index))

        show_plots = False
        os.chdir(home)
        hdf5_file = tables.open_file('attstrac_specific.hdf5', mode='r')
        xuv = hdf5_file.root.xuv_real[index, :] + 1j * hdf5_file.root.xuv_imag[index, :]
        trace = hdf5_file.root.trace[index, :].reshape(len(p_vec), len(tauvec))
        hdf5_file.close()

        # change to plotting directory
        if not os.path.isdir('./plotting'):
            print('creating plotting folder')
            os.makedirs('./plotting')

        os.chdir('./plotting/')

        trace_filtered_time, params = construct_proof(trace, tauvec, dt, f0_ir)

        tauvec_f_space = params['tauvec_f_space']
        traceft =  params['traceft']
        filter2d = params['filter2d']
        filter = params['filter']
        trace_filtered = params['trace_filtered']
        tauvec_time = params['tauvec_time']

        # spectrum of xuv for normalization of image
        K = (0.5 * p_vec ** 2).reshape(-1, 1)
        e_fft = np.exp(-1j * (K + items['Ip']) * xuv_int_t.reshape(1, -1))
        product = xuv.reshape(1, -1) * e_fft
        integral = np.abs(dt * np.sum(product, axis=1)) ** 2

        # need to figure out this scaling later
        offset = 0.1
        compensation = 1 / (integral + offset)
        # compensation = 1 + scaling * (np.max(integral) - integral)

        xuv_spectrum_compensation = np.ones_like(trace) * compensation.reshape(-1, 1)

        trace_compensated = np.abs(trace_filtered_time) * xuv_spectrum_compensation

        plot_before_after_filter()

        plot_before_after_compensation()

        plot_ir_xuv_atto()

        plot_filtering()

        plot_compensation()

        plot_xuv_att_proof()

        if show_plots:
            plt.show()

        plt.close('all')

