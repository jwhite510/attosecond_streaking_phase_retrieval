from crab_tf2 import *
import tables
import scipy.constants as sc





def add_shot_noise(trace_sample):

    # set the maximum counts for the image
    counts_min, counts_max = 50, 200
    # maximum counts between two values
    counts = np.round(counts_min + np.random.rand() * (counts_max - counts_min))

    discrete_trace = np.round(trace_sample*counts)

    noise = np.random.poisson(lam=discrete_trace) - discrete_trace

    noisy_trace = discrete_trace + noise

    noisy_trace_normalized = noisy_trace / np.max(noisy_trace)

    return noisy_trace_normalized



def plot_opened_file(xuv_t, ir_t, trace):


    fig = plt.figure()
    gs = fig.add_gridspec(2,2)

    # plot xuv
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(xuv.tmat, np.real(xuv_t), color='blue')

    # plot ir
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ir.tmat, np.real(ir_t), color='blue')

    # plot trace
    ax = fig.add_subplot(gs[1,:])
    ax.pcolormesh(tau_values, p_values, trace.reshape(len(p_values), len(tau_values)), cmap='jet')



def generate_processed_traces(filename, coefs):

    # create a file for proof traces and xuv envelopes
    processed_filename = filename.split('.')[0] +'_processed.hdf5'
    print('creating file: '+processed_filename)

    with tables.open_file(processed_filename, mode='w') as processed_data:

        processed_data.create_earray(processed_data.root, 'trace', tables.Float64Atom(), shape=(0, len(p_values) * len(tau_values)))

        processed_data.create_earray(processed_data.root, 'coefs', tables.Float64Atom(), shape=(0, int(coefs)))

        processed_data.create_earray(processed_data.root, 'ir_t', tables.ComplexAtom(itemsize=32), shape=(0, len(ir.tmat)))

        processed_data.create_earray(processed_data.root, 'ir_f', tables.ComplexAtom(itemsize=32), shape=(0, len(ir.Ef_prop_cropped)))

        processed_data.create_earray(processed_data.root, 'xuv_f', tables.ComplexAtom(itemsize=32),
                              shape=(0, len(xuv.Ef_prop_cropped)))

        processed_data.create_earray(processed_data.root, 'xuv_t', tables.ComplexAtom(itemsize=32), shape=(0, len(xuv.tmat)))

        ## create new array for processed sample here
        # processed_data.create_earray(processed_data.root, 'processed', tables.ComplexAtom(itemsize=32),shape=(0, len(xuv.tmat)))


        processed_data.close()


    with tables.open_file(filename, mode='r') as unprocessed_datafile:
        with tables.open_file(processed_filename, mode='a') as processed_data:

            # get the number of data points
            samples = np.shape(unprocessed_datafile.root.xuv_t[:, :])[0]
            print('Samples to be processed: ', samples)

            for index in range(samples):

                if index % 500 == 0:
                    print(index, '/', samples)

                xuv_f_sample = unprocessed_datafile.root.xuv_f[index, :]
                xuv_t_sample = unprocessed_datafile.root.xuv_t[index, :]

                ir_t_sample = unprocessed_datafile.root.ir_t[index, :]
                ir_f_sample = unprocessed_datafile.root.ir_f[index, :]

                trace_sample = unprocessed_datafile.root.trace[index, :].reshape(len(p_values), len(tau_values))

                coefs_sample = unprocessed_datafile.root.coefs[index, :]

                # PROCESS DATA HERE
                trace_sample = add_shot_noise(trace_sample)
                # processed_trace / sample = process(trace_sample)
                # divide the xuv_f by modulus
                #plt.figure(9998)
                #plt.plot(np.real(xuv_f_sample), color='red')
                #plt.plot(np.imag(xuv_f_sample), color='blue')

                # for just the phase, not modulus
                #xuv_f_sample = xuv_f_sample / np.abs(xuv_f_sample)

                #plt.figure(9999)
                #plt.plot(np.real(xuv_f_sample), color='red')
                #plt.plot(np.imag(xuv_f_sample), color='blue')
                #plt.ioff()
                #plt.show()
                #exit(0)


                # append the data to the processed hdf5 file
                processed_data.root.xuv_f.append(xuv_f_sample.reshape(1, -1))
                processed_data.root.xuv_t.append(xuv_t_sample.reshape(1, -1))

                processed_data.root.ir_t.append(ir_t_sample.reshape(1, -1))
                processed_data.root.ir_f.append(ir_f_sample.reshape(1, -1))

                processed_data.root.trace.append(trace_sample.reshape(1, -1))

                processed_data.root.coefs.append(coefs_sample.reshape(1, -1))




if __name__ == "__main__":


    # unprocessed_filename = 'attstrac_specific.hdf5'
    generate_processed_traces(filename='attstrace_train2.hdf5', coefs=5)
    generate_processed_traces(filename='attstrace_test2.hdf5', coefs=5)

    # test opening the file
    index = 20
    test_open_file = 'attstrace_train2_processed.hdf5'
    # test_open_file = 'attstrace_test2_processed.hdf5'
    with tables.open_file(test_open_file, mode='r') as processed_data:

        xuv_t = processed_data.root.xuv_t[index, :]
        ir_t = processed_data.root.ir_t[index, :]
        trace = processed_data.root.trace[index, :]

    plot_opened_file(xuv_t, ir_t, trace)

    plt.show()




