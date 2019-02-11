import tables
import tensorflow as tf
import tf_functions
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import xuv_spectrum.spectrum
import ir_spectrum.ir_spectrum
import time


def plot_opened_file(xuv_coefs, ir_params, trace, sess, tf_graphs, streak_params):

    fig = plt.figure()
    gs = fig.add_gridspec(2, 2)

    print("xuv_coefs: ", xuv_coefs)
    print("ir_params: ", ir_params)

    xuv_t = sess.run(tf_graphs["xuv_E_prop"]["t"], feed_dict={tf_graphs["xuv_coefs_in"]: xuv_coefs.reshape(1, -1)})

    ir_t = sess.run(tf_graphs["ir_E_prop"]["t"], feed_dict={tf_graphs["ir_values_in"]: ir_params.reshape(1, -1)})

    # plot xuv
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(xuv_spectrum.spectrum.tmat, np.real(xuv_t[0]), color='blue')

    # plot ir
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ir_spectrum.ir_spectrum.tmat, np.real(ir_t[0]), color='blue')


    # plot trace
    ax = fig.add_subplot(gs[1, :])
    ax.pcolormesh(streak_params["tau_values"], streak_params["p_values"],
                  trace.reshape(len(streak_params["p_values"]), len(streak_params["tau_values"])), cmap='jet')


def update_plots2(axes, trace, xuv_t, threshold):

    axes[0].cla()
    axes[0].pcolormesh(trace, cmap='jet')
    axes[1].cla()
    #    ax[1].plot(xuv.tmat, np.real(xuv.Et_prop), color='blue')
    axes[1].plot(np.real(xuv_t), color='blue')
    maxval = np.max(np.abs(xuv_t))

    axes[1].plot([threshold["indexes"][0], threshold["indexes"][0]], [maxval, -maxval], color='black')
    axes[1].plot([threshold["indexes"][1], threshold["indexes"][1]], [maxval, -maxval], color='black')
    axes[1].plot([threshold["indexes"][0], threshold["indexes"][1]], [threshold["threshold"], threshold["threshold"]], color='red', linestyle='dashed')

    plt.pause(0.001)


def check_time_boundary(indexmin, indexmax, threshold, xuv_t, bad_samples):
    # check to make sure the signal decreases in time
    # indexmin = 100
    # indexmax = 924
    value_1 = np.abs(xuv_t)[indexmin]
    value_2 = np.abs(xuv_t)[indexmax]


    if value_1 > threshold or value_2 > threshold:
    # if True:
        bad_samples+=1
        # plt.figure(356)
        # plt.cla()
        # plt.plot(np.real(xuv_t))
        # plt.plot([indexmin, indexmin], [-np.max(np.real(xuv_t)), np.max(np.real(xuv_t))], color='black')
        # plt.plot([indexmax, indexmax], [-np.max(np.real(xuv_t)), np.max(np.real(xuv_t))], color='black')
        # plt.plot([indexmin, indexmax], [threshold, threshold], color='red', linestyle='dashed')
        # plt.text(0.1, 0.8, 'bad samples:{}'.format(bad_samples), transform=plt.gca().transAxes)
        # plt.show()
        # plt.pause(0.001)
        return bad_samples, False

    return bad_samples, True


def generate_samples(tf_graphs, n_samples, filename, streak_params, xuv_coefs, sess, axis):
    print('creating file: ' + filename)

    # create hdf5 file
    with tables.open_file(filename, mode='w') as hd5file:
        # create array for trace
        hd5file.create_earray(hd5file.root, 'trace', tables.Float64Atom(), shape=(0, len(streak_params["p_values"]) * len(streak_params["tau_values"])))

        # noise trace
        hd5file.create_earray(hd5file.root, 'noise_trace', tables.Float64Atom(),shape=(0, len(streak_params["p_values"]) * len(streak_params["tau_values"])))

        # create array for XUV
        hd5file.create_earray(hd5file.root, 'xuv_coefs', tables.Float64Atom(), shape=(0, xuv_coefs))

        # create array for IR
        hd5file.create_earray(hd5file.root, 'ir_params', tables.Float64Atom(), shape=(0, 4))

        hd5file.close()


    # make a sample with no phase to give a comparison
    xuv_coefs_in = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    xuv_t = sess.run(tf_graphs["xuv_E_prop"]["t"], feed_dict={tf_graphs["xuv_coefs_in"]: xuv_coefs_in})
    threshold = np.max(np.abs(xuv_t)) / 300
    indexmin = 100
    indexmax = 924

    threshold_dict = {}
    threshold_dict["threshold"] = threshold
    threshold_dict["indexes"] = (indexmin, indexmax)

    # count the number of bad samples generated
    bad_samples = 0


    # open and append the file
    with tables.open_file(filename, mode='a') as hd5file:
        for i in range(n_samples):

            xuv_good = False

            while not xuv_good:

                #xuv_coefs_in = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])

                #random coefficients between -0.5 and 0.5
                xuv_coefs_rand = (np.random.rand(4)-0.5).reshape(1, -1)
                xuv_coefs_in = np.append(np.array([[0.0]]), xuv_coefs_rand, axis=1)

                # generate time pulse from these coefficients
                xuv_t = sess.run(tf_graphs["xuv_E_prop"]["t"], feed_dict={tf_graphs["xuv_coefs_in"]: xuv_coefs_in})

                # check if the pulse is constrained in time window
                bad_samples, xuv_good = check_time_boundary(indexmin, indexmax, threshold, xuv_t[0], bad_samples)


            # make ir params
            ir_values_in = (2.0*np.random.rand(4)-1.0).reshape(1, -1)

            # generate streaking trace
            if i % 500 == 0:
                print('generating sample {} of {}'.format(i + 1, n_samples))
                print('bad sample count: {}'.format(bad_samples))
                # generate the streaking trace
                time1 = time.time()

                trace = sess.run(tf_graphs["image"], feed_dict={tf_graphs["xuv_coefs_in"]: xuv_coefs_in,
                                                                tf_graphs["ir_values_in"]: ir_values_in})

                update_plots2(axes=axis, trace=trace, xuv_t=xuv_t[0], threshold=threshold_dict)


                time2 = time.time()
                duration = time2 - time1
                print('duration: {} s'.format(round(duration, 4)))

            else:
                trace = sess.run(tf_graphs["image"], feed_dict={tf_graphs["xuv_coefs_in"]: xuv_coefs_in,
                                                                tf_graphs["ir_values_in"]: ir_values_in})


            # add noise to trace
            noise_trace = add_shot_noise(trace)


            # append data sample
            hd5file.root.trace.append(trace.reshape(1, -1))
            hd5file.root.noise_trace.append(noise_trace.reshape(1, -1))
            hd5file.root.xuv_coefs.append(xuv_coefs_in.reshape(1, -1))
            hd5file.root.ir_params.append(ir_values_in.reshape(1, -1))


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




if __name__ == "__main__":

    # initialize XUV generator
    xuv_phase_coeffs = 5
    xuv_coefs_in = tf.placeholder(tf.float32, shape=[None, xuv_phase_coeffs])
    xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_in, amplitude=12.0)

    # initialize IR generator
    # IR amplitudes
    amplitudes = {}
    amplitudes["phase_range"] = (0, 2 * np.pi)
    amplitudes["clambda_range"] = (1.6345, 1.6345)
    amplitudes["pulseduration_range"] =  (7.0, 12.0)
    amplitudes["I_range"] = (0.4, 1.0)
    # IR creation
    ir_values_in = tf.placeholder(tf.float32, shape=[None, 4])
    ir_E_prop = tf_functions.ir_from_params(ir_values_in, amplitudes=amplitudes)

    # initialize streaking trace generator
    # Neon
    Ip_eV = 21.5645
    Ip = Ip_eV * sc.electron_volt  # joules
    Ip = Ip / sc.physical_constants['atomic unit of energy'][0]  # a.u.

    # construct streaking image
    image, streak_params = tf_functions.streaking_trace(xuv_cropped_f_in=xuv_E_prop["f_cropped"][0],
                                         ir_cropped_f_in=ir_E_prop["f_cropped"][0],Ip=Ip)


    tf_graphs = {}
    tf_graphs["xuv_coefs_in"] = xuv_coefs_in
    tf_graphs["ir_values_in"] = ir_values_in
    tf_graphs["xuv_E_prop"] = xuv_E_prop
    tf_graphs["ir_E_prop"] = ir_E_prop
    tf_graphs["image"] = image

    # create plot to show samples as they are generated
    _, ax = plt.subplots(2, 1, figsize=(5, 5))
    with tf.Session() as sess:


        generate_samples(tf_graphs=tf_graphs, n_samples=60000,
                         filename="train3.hdf5", streak_params=streak_params,
                         xuv_coefs=xuv_phase_coeffs, sess=sess, axis=ax)

        generate_samples(tf_graphs=tf_graphs, n_samples=500,
                         filename="test3.hdf5", streak_params=streak_params,
                         xuv_coefs=xuv_phase_coeffs, sess=sess, axis=ax)


        # test open the file
        index = 2
        with tables.open_file('train3.hdf5', mode='r') as hd5file:
            xuv_coefs = hd5file.root.xuv_coefs[index, :]
            ir_params = hd5file.root.ir_params[index, :]
            trace = hd5file.root.noise_trace[index, :]

        plot_opened_file(xuv_coefs=xuv_coefs, ir_params=ir_params,
                         trace=trace, sess=sess, tf_graphs=tf_graphs,
                         streak_params=streak_params)

        plt.ioff()
        plt.show()




