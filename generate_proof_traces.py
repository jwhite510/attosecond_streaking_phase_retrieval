import pickle
import numpy as np
import matplotlib.pyplot as plt
import tables
from scipy import interpolate
from proof import construct_proof
import matplotlib.patches as mpatches
from generate_data import XUV_Field_rand_phase


def plot_elements(trace, proof_trace, xuv, xuv_envelope, xuv_frequency):

    fig = plt.figure(constrained_layout=False, figsize=(7, 5))
    gs = fig.add_gridspec(3, 2)

    ax = fig.add_subplot(gs[0, 0])
    ax.pcolormesh(trace, cmap='jet')
    ax.text(0, 0.9, '1', transform=ax.transAxes, backgroundcolor='yellow')
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 1])
    ax.pcolormesh(proof_trace, cmap='jet')
    ax.text(0, 0.9, '2', transform=ax.transAxes, backgroundcolor='yellow')
    ax.set_xticks([])
    ax.set_yticks([])


    ax = fig.add_subplot(gs[1, 0])
    ax.plot(xuv_int_t, np.real(xuv), color='blue')
    ax.plot(xuv_int_t, np.imag(xuv), color='red')
    ax.plot(xuv_int_t, np.abs(xuv), color='black', linestyle='dashed')
    ax.text(0, 0.9, '1', transform=ax.transAxes, backgroundcolor='yellow')

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(time_reduced_dim, np.real(xuv_envelope), color='blue')
    ax.plot(time_reduced_dim, np.imag(xuv_envelope), color='red')
    ax.plot(time_reduced_dim, np.abs(xuv_envelope), color='black', linestyle='dashed')
    ax.text(0, 0.9, '2', transform=ax.transAxes, backgroundcolor='yellow')
    ax.plot([time_reduced_dim[0], time_reduced_dim[-1]], [0, 0], color='black', alpha=0.5)
    ax.plot([0, 0], [0.2, -0.2], color='black', alpha=0.5)


    ax = fig.add_subplot(gs[2, 1])
    ax.plot(frequency_axis, np.real(xuv_frequency), color='blue')
    ax.plot(frequency_axis, np.imag(xuv_frequency), color='red')





    plt.savefig('./processdata.png')
    plt.show()


def process_xuv(xuv, xuv_time, f0, reduction, plotting=True):
    global time_reduced_dim
    f0_removed = xuv * np.exp(-1j * 2 * np.pi * f0 * xuv_time)
    t_steps_reduced = len(xuv)/reduction
    time_reduced_dim = np.linspace(xuv_time[0], xuv_time[-1], t_steps_reduced)
    f = interpolate.interp1d(xuv_time, f0_removed, kind='cubic')
    xuv_reduced = f(time_reduced_dim)


    if plotting:
        fig, ax = plt.subplots(3, 1, figsize=(7, 10))

        ax[0].plot(xuv_time, np.real(xuv), color='blue')
        ax[0].plot(xuv_time, np.imag(xuv), color='red')
        ax[0].plot(xuv_time, np.abs(xuv), color='orange', linestyle='dashed')

        ax[1].plot(xuv_time, np.real(f0_removed), color='blue')
        ax[1].plot(xuv_time, np.imag(f0_removed), color='red')
        ax[1].plot(xuv_time, np.abs(f0_removed), color='orange', linestyle='dashed')

        axtwin = ax[1].twinx()
        axtwin.plot(xuv_time, np.unwrap(np.angle(f0_removed)), color='green', alpha=0.5)

        ax[2].plot(time_reduced_dim, np.real(xuv_reduced), color='blue')
        ax[2].plot(time_reduced_dim, np.imag(xuv_reduced), color='red')

        plt.show()

    return xuv_reduced




def generate_processed_traces(filename):

    # create a file for proof traces and xuv envelopes

    processed_filename = filename.split('.')[0] +'_processed.hdf5'
    print('creating file: '+processed_filename)

    with tables.open_file(processed_filename, mode='w') as processed_data:

        processed_data.create_earray(processed_data.root, 'attstrace', tables.Float16Atom(),
                                     shape=(0, len(p_vec) * len(tauvec)))

        processed_data.create_earray(processed_data.root, 'proof', tables.Float16Atom(),
                                     shape=(0, len(p_vec) * len(tauvec)))

        processed_data.create_earray(processed_data.root, 'xuv', tables.ComplexAtom(itemsize=16),
                                     shape=(0, len(xuv_int_t)))

        processed_data.create_earray(processed_data.root, 'xuv_envelope', tables.ComplexAtom(itemsize=16),
                                     shape=(0, int(len(xuv_int_t) / xuv_dimmension_reduction)))

        processed_data.create_earray(processed_data.root, 'xuv_frequency_domain', tables.ComplexAtom(itemsize=16),
                                     shape=(0, 64))




    with tables.open_file(filename, mode='r') as unprocessed_datafile:
        with tables.open_file(processed_filename, mode='a') as processed_data:

            # get the number of data points
            samples = np.shape(unprocessed_datafile.root.xuv_real[:, :])[0]
            print('Samples to be processed: ', samples)

            for index in range(samples):

                if index % 5 == 0:
                    print(index, '/', samples)

                xuv = unprocessed_datafile.root.xuv_real[index, :] + 1j * unprocessed_datafile.root.xuv_imag[index, :]
                attstrace = unprocessed_datafile.root.trace[index, :].reshape(len(p_vec), len(tauvec))


                # construct proof trace
                proof_trace, _ = construct_proof(attstrace, tauvec=tauvec, dt=dt, f0_ir=f0_ir)

                # construct xuv pulse minus central oscilating term
                xuv_envelope = process_xuv(xuv, xuv_time=xuv_int_t, f0=xuvf0,
                                           reduction=xuv_dimmension_reduction, plotting=False)

                # retrieve the xuv in frequency domain
                xuv_frequency_d = unprocessed_datafile.root.xuv_frequency_domain[index, :]

                # plot_elements(attstrace, np.real(proof_trace), xuv, xuv_envelope)

                # append the data to the processed hdf5 file
                processed_data.root.xuv_frequency_domain.append(xuv_frequency_d.reshape(1, -1))
                processed_data.root.attstrace.append(attstrace.reshape(1, -1))
                processed_data.root.proof.append(np.real(proof_trace).reshape(1, -1))
                processed_data.root.xuv.append(xuv.reshape(1, -1))
                processed_data.root.xuv_envelope.append(xuv_envelope.reshape(1, -1))

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
    xuvf0 = crab_tf_items['xuvf0']


except Exception as e:
    print(e)
    print('run crab_tf.py first to pickle the needed files')
    exit(0)

xuv_dimmension_reduction = 8
xuv_field_length = int(len(xuv_int_t)/xuv_dimmension_reduction)

if __name__ == "__main__":

    # get the frequency axis
    xuv_test = XUV_Field_rand_phase(phase_amplitude=0, phase_nodes=100, plot=False)
    frequency_axis = xuv_test.f_cropped_cropped

    print('Dimmension reduction: ', xuv_dimmension_reduction)
    print('xuv field length: ', int(len(xuv_int_t)/xuv_dimmension_reduction))

    # unprocessed_filename = 'attstrac_specific.hdf5'
    generate_processed_traces(filename='attstrace_train.hdf5')
    generate_processed_traces(filename='attstrace_test.hdf5')

    # test opening the file
    index = 0
    # test_open_file = 'attstrace_test_processed.hdf5'
    test_open_file = 'attstrace_train_processed.hdf5'
    with tables.open_file(test_open_file, mode='r') as processed_data:
        xuv1 = processed_data.root.xuv[index, :]
        xuv_envelope1 = processed_data.root.xuv_envelope[index, :]
        attstrace1 = processed_data.root.attstrace[index, :].reshape(len(p_vec), len(tauvec))
        proof_trace1 = processed_data.root.proof[index, :].reshape(len(p_vec), len(tauvec))
        xuv_f = processed_data.root.xuv_frequency_domain[index, :]

        plot_elements(attstrace1, np.real(proof_trace1), xuv1, xuv_envelope1, xuv_f)















