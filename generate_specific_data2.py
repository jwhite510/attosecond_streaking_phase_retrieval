from crab_tf2 import *
import tables




def plot_elements(xuv_f, xuv_t, ir_f, ir_t, trace):

    fig = plt.figure()
    gs = fig.add_gridspec(3,2)

    # plot streaking trace
    ax = fig.add_subplot(gs[0,:])
    ax.pcolormesh(trace, cmap='jet')

    # plot xuv f
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(np.real(xuv_f), color='blue')
    ax.plot(np.imag(xuv_f), color='red')

    # plot xuv t
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(np.real(xuv_t), color='blue')
    ax.plot(np.imag(xuv_t), color='red')

    # plot ir f
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(np.real(ir_f), color='blue')
    ax.plot(np.imag(ir_f), color='red')

    # plot ir t
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(np.real(ir_t), color='blue')
    ax.plot(np.imag(ir_t), color='red')



def create_hdf5(filename):

    print('creating file: ' + filename)

    # create hdf5 file
    with tables.open_file(filename, mode='w') as hd5file:

        hd5file.create_earray(hd5file.root,'trace', tables.Float64Atom(), shape=(0, len(p_values) * len(tau_values)))

        hd5file.create_earray(hd5file.root,'ir_t', tables.ComplexAtom(itemsize=32), shape=(0, len(ir.tmat)))

        hd5file.create_earray(hd5file.root,'ir_f', tables.ComplexAtom(itemsize=32), shape=(0, len(ir.Ef_prop_cropped)))

        hd5file.create_earray(hd5file.root, 'xuv_f', tables.ComplexAtom(itemsize=32), shape=(0, len(xuv.Ef_prop_cropped)))

        hd5file.create_earray(hd5file.root, 'xuv_t', tables.ComplexAtom(itemsize=32), shape=(0, len(xuv.tmat)))

        hd5file.close()





if __name__ == "__main__":

    specific_traces = {'xuv': [], 'ir': []}


    xuv = XUV_Field(tod=8000, gdd=500)

    # trace 1
    ir = IR_Field()
    specific_traces['xuv'].append(xuv)
    specific_traces['ir'].append(ir)

    # trace 2
    ir = IR_Field(random_pulse={'phase_range':(0,2*np.pi), 'clambda_range': (1.2,2.3), 'pulse_duration_range':(7.0,12.0)})
    specific_traces['xuv'].append(xuv)
    specific_traces['ir'].append(ir)

    # trace 3
    ir = IR_Field(random_pulse={'phase_range': (0, 2 * np.pi), 'clambda_range': (1.2, 2.3), 'pulse_duration_range': (7.0, 12.0)})
    specific_traces['xuv'].append(xuv)
    specific_traces['ir'].append(ir)


    # create the file
    filename = 'attstrac_specific.hdf5'
    create_hdf5(filename)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        init.run()

        # open the hdf5 file
        with tables.open_file(filename, mode='a') as hdf5file:

            for i in range(len(specific_traces['xuv'])):

                hdf5file.root.xuv_t.append(specific_traces['xuv'][i].Et_prop.reshape(1,-1))
                hdf5file.root.xuv_f.append(specific_traces['xuv'][i].Ef_prop_cropped.reshape(1,-1))

                hdf5file.root.ir_t.append(specific_traces['ir'][i].Et_prop.reshape(1,-1))
                hdf5file.root.ir_f.append(specific_traces['ir'][i].Ef_prop_cropped.reshape(1,-1))

                trace = sess.run(image, feed_dict={ir_cropped_f: specific_traces['ir'][i].Ef_prop_cropped, xuv_cropped_f: specific_traces['xuv'][i].Ef_prop_cropped})

                hdf5file.root.trace.append(trace.reshape(1,-1))


    # open and view the trace
    index = 2
    with tables.open_file(filename, mode='r') as hdf5file:

        xuv_f = hdf5file.root.xuv_f[index, :]
        xuv_t = hdf5file.root.xuv_t[index, :]

        ir_t = hdf5file.root.ir_t[index, :]
        ir_f = hdf5file.root.ir_f[index, :]

        trace = hdf5file.root.trace[index, :].reshape(len(p_values), len(tau_values))

        plot_elements(xuv_f, xuv_t, ir_f, ir_t, trace)

        plt.show()






























