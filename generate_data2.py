from crab_tf2 import *
import tables




def check_time_boundary(indexmin, indexmax, threshold, xuv):
    global bad_samples
    # check to make sure the signal decreases in time
    # indexmin = 100
    # indexmax = 924
    value_1 = np.abs(xuv.Et_prop)[indexmin]
    value_2 = np.abs(xuv.Et_prop)[indexmax]


    if value_1 > threshold or value_2 > threshold:
        bad_samples+=1
        #plt.figure(356)
        #plt.cla()
        #plt.plot(np.real(xuv.Et_prop))
        #plt.plot([indexmin, indexmin], [-np.max(np.real(xuv.Et_prop)), np.max(np.real(xuv.Et_prop))], color='black')
        #plt.plot([indexmax, indexmax], [-np.max(np.real(xuv.Et_prop)), np.max(np.real(xuv.Et_prop))], color='black')
        #plt.plot([indexmin, indexmax], [threshold, threshold], color='red', linestyle='dashed')
        #plt.text(0.1, 0.8, 'bad samples:{}'.format(bad_samples), transform=plt.gca().transAxes)
        #plt.show()
        #plt.pause(0.001)
        return False

    return True






def update_plots(axes, xuv, ir, trace, threshold, threshindexes):

    ax[0].cla()
    ax[0].pcolormesh(trace, cmap='jet')
    ax[1].cla()
#    ax[1].plot(xuv.tmat, np.real(xuv.Et_prop), color='blue')
    ax[1].plot(np.real(xuv.Et_prop), color='blue')
    maxval = np.max(np.abs(xuv.Et_prop))
    ax[1].plot([threshindexes[0], threshindexes[0]], [maxval, -maxval], color='black')
    ax[1].plot([threshindexes[1], threshindexes[1]], [maxval, -maxval], color='black')
    ax[1].plot([threshindexes[0], threshindexes[1]], [threshold, threshold], color='red', linestyle='dashed')


    plt.pause(0.001)


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
            #xuv_sample = XUV_Field(random_phase={'nodes': 50, 'amplitude': 6}, measured_spectrum=spectrum_data)

            #plt.figure(999)
            #plt.plot(np.real(xuv_sample.Et_prop))
            #plt.ioff()
            #plt.show()
            #exit(0)




            xuv_good = False
            threshold = np.max(np.abs(xuv.Et_prop)) / 100
            indexmin = 100
            indexmax = 924
            while not xuv_good:

                # random with only taylor coefs
                xuv_sample = XUV_Field(random_phase_taylor={'coefs': 5, 'amplitude': 12},
                               measured_spectrum=spectrum_data)

                #xuv sample to match the measured spectrum, but with gaussian
                #xuv_sample = XUV_Field(random_phase_taylor={'coefs': 3, 'amplitude': 5},
                #                       f0=10.0e16, tmax=8e-16, N=1024,
                #                       start_index=spectrum_data['indexmin'],
                #                       end_index=spectrum_data['indexmax'])

                #plt.figure(333)
                #plt.plot(xuv_sample.fmat, np.zeros_like(xuv_sample.fmat))
                #plt.plot([xuv_sample.taylor_f0, xuv_sample.taylor_f0], [-1, 1])
                #plt.plot(xuv_sample.fmat, np.abs(xuv_sample.Ef))
                #plt.ioff()
                #plt.show()
                #exit(0)


                # xuv sample to match the measured spectrum, but with gaussian
                #xuv_sample = XUV_Field(random_phase={'nodes': 100, 'amplitude': 6},
                #                       f0=10.0e16, tmax=8e-16, N=1024,
                #                       start_index=spectrum_data['indexmin'],
                #                       end_index=spectrum_data['indexmax'])

                xuv_good = check_time_boundary(indexmin, indexmax, threshold, xuv_sample)






                #xuv_sample = XUV_Field(random_phase_taylor={'coefs': 20, 'amplitude': 200})


            #plt.figure(444)
            #plt.plot(np.real(xuv_sample.Et_prop))
            #plt.ioff()
            #plt.show()
            #exit(0)

            # generate a random IR pulse
            #ir_sample = IR_Field(random_pulse={'phase_range':(0,2*np.pi),
            #                             'clambda_range': (1.2,2.3),
            #                             'pulse_duration_range':(7.0,12.0),
            #                                   'I_range': (0.05, 0.6)})


            # generate IR pulses with random phase only
            #ir_sample = IR_Field(random_pulse={'phase_range':(0,2*np.pi),
            #                            'clambda_range': (1.7,1.7),
            #                            'pulse_duration_range':(12.0,12.0),
            #                                  'I_range': (1.0, 1.0)})

            # generate IR pulses with random phase, pulse duration, inensity
            ir_sample = IR_Field(random_pulse={'phase_range': (0, 2 * np.pi),
                                               'clambda_range': (1.6345, 1.6345),
                                               'pulse_duration_range': (7.0, 12.0),
                                               'I_range': (0.4, 1.0)})
            # print(ir_sample.f0)
            # lam0 = sc.c / (ir_sample.f0 / sc.physical_constants['atomic unit of time'][0])
            # print(lam0)

            # generate a default IR pulse
            #ir_sample = IR_Field()

            # generate the streaking trace
            if i % 500 == 0:
                print('generating sample {} of {}'.format(i + 1, n_samples))
                print('bad sample count: {}'.format(bad_samples))
                # generate the FROG trace
                time1 = time.time()

                strace = sess.run(image,feed_dict={
                                                ir_cropped_f: ir_sample.Ef_prop_cropped,
                                                xuv_cropped_f: xuv_sample.Ef_prop_cropped})
                update_plots(axes=ax, xuv=xuv_sample, ir=ir_sample, trace=strace,
                             threshold=threshold, threshindexes=(indexmin, indexmax))


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
    bad_samples = 0

    plt.ion()
    _, ax = plt.subplots(2,1, figsize=(5,5))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        n_train_samples = 80000
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











