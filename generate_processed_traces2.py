from crab_tf2 import *
import tables


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



def generate_processed_traces(filename):

    # create a file for proof traces and xuv envelopes
    processed_filename = filename.split('.')[0] +'_processed.hdf5'
    print('creating file: '+processed_filename)

    with tables.open_file(processed_filename, mode='w') as processed_data:

        processed_data.create_earray(processed_data.root, 'trace', tables.Float64Atom(), shape=(0, len(p_values) * len(tau_values)))

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

                # PROCESS DATA HERE
                # processed_trace / sample = process(trace_sample)


                # append the data to the processed hdf5 file
                processed_data.root.xuv_f.append(xuv_f_sample.reshape(1, -1))
                processed_data.root.xuv_t.append(xuv_t_sample.reshape(1, -1))

                processed_data.root.ir_t.append(ir_t_sample.reshape(1, -1))
                processed_data.root.ir_f.append(ir_f_sample.reshape(1, -1))

                processed_data.root.trace.append(trace_sample.reshape(1, -1))




if __name__ == "__main__":


    # unprocessed_filename = 'attstrac_specific.hdf5'
    generate_processed_traces(filename='attstrace_train2.hdf5')
    generate_processed_traces(filename='attstrace_test2.hdf5')

    # test opening the file
    index = 0
    test_open_file = 'attstrace_train2_processed.hdf5'
    # test_open_file = 'attstrace_test2_processed.hdf5'
    with tables.open_file(test_open_file, mode='r') as processed_data:

        xuv_t = processed_data.root.xuv_t[index, :]
        ir_t = processed_data.root.ir_t[index, :]
        trace = processed_data.root.trace[index, :]

    plot_opened_file(xuv_t, ir_t, trace)

    plt.show()



