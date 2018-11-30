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



def generate_samples(n_samples, filename):
    print('creating file: ' + filename)

    # create hdf5 file
    with tables.open_file(filename, mode='w') as hd5file:
        hd5file.create_earray(hd5file.root, 'trace', tables.Float64Atom(), shape=(0, len(p_values) * len(tau_values)))

        hd5file.create_earray(hd5file.root, 'ir_t', tables.ComplexAtom(itemsize=32), shape=(0, len(ir.tmat)))

        hd5file.create_earray(hd5file.root, 'ir_f', tables.ComplexAtom(itemsize=32), shape=(0, len(ir.Ef_prop_cropped)))

        hd5file.create_earray(hd5file.root, 'xuv_f', tables.ComplexAtom(itemsize=32),
                              shape=(0, len(xuv.Ef_prop_cropped)))

        hd5file.create_earray(hd5file.root, 'xuv_t', tables.ComplexAtom(itemsize=32), shape=(0, len(xuv.tmat)))

        hd5file.close()


    # open and append the file
    with tables.open_file(filename, mode='a') as hd5file:

        for i in range(n_samples):

            # generate a random xuv pulse
            xuv_sample = XUV_Field(random_phase={'nodes': 100, 'amplitude': 6})

            # generate a random IR pulse
            ir_sample = IR_Field(random_pulse={'phase_range':(0,2*np.pi),
                                         'clambda_range': (1.2,2.3),
                                         'pulse_duration_range':(7.0,12.0)})

            # generate a default IR pulse
            #ir_sample = IR_Field()

            # generate the streaking trace
            if i % 500 == 0:
                print('generating sample {} of {}'.format(i + 1, n_samples))
                # generate the FROG trace
                time1 = time.time()

                strace = sess.run(image,feed_dict={
                                                ir_cropped_f: ir_sample.Ef_prop_cropped,
                                                xuv_cropped_f: xuv_sample.Ef_prop_cropped})


                time2 = time.time()
                duration = time2 - time1
                print('duration: {} s'.format(round(duration, 4)))

            else:
                strace = sess.run(image, feed_dict={
                                                    ir_cropped_f: ir_sample.Ef_prop_cropped,
                                                    xuv_cropped_f: xuv_sample.Ef_prop_cropped})



            # append xuv time and frequency
            hd5file.root.xuv_t.append(xuv_sample.Et_prop.reshape(1, -1))
            hd5file.root.xuv_f.append(xuv_sample.Ef_prop_cropped.reshape(1, -1))

            # append ir time and frequency
            hd5file.root.ir_t.append(ir_sample.Et_prop.reshape(1, -1))
            hd5file.root.ir_f.append(ir_sample.Ef_prop_cropped.reshape(1, -1))

            # append trace
            hd5file.root.trace.append(strace.reshape(1, -1))



if __name__ == "__main__":

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        n_train_samples = 20000
        n_test_samples = 500

        generate_samples(n_samples=n_train_samples, filename='attstrace_train2.hdf5')
        generate_samples(n_samples=n_test_samples, filename='attstrace_test2.hdf5')



    # test open the file
    index = 20
    with tables.open_file('attstrace_train2.hdf5', mode='r') as hd5file:

        xuv_t = hd5file.root.xuv_t[index, :]
        ir_t = hd5file.root.ir_t[index, :]
        trace = hd5file.root.trace[index, :]


    plot_opened_file(xuv_t, ir_t, trace)

    plt.show()











