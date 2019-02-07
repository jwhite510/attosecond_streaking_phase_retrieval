import tables
import tensorflow as tf
import tf_functions
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import xuv_spectrum.spectrum



def check_time_boundary(indexmin, indexmax, threshold, xuv_t, bad_samples):
    # check to make sure the signal decreases in time
    # indexmin = 100
    # indexmax = 924
    value_1 = np.abs(xuv_t)[indexmin]
    value_2 = np.abs(xuv_t)[indexmax]


    if value_1 > threshold or value_2 > threshold:
    # if True:
        bad_samples+=1
        plt.figure(356)
        plt.cla()
        plt.plot(np.real(xuv_t))
        plt.plot([indexmin, indexmin], [-np.max(np.real(xuv_t)), np.max(np.real(xuv_t))], color='black')
        plt.plot([indexmax, indexmax], [-np.max(np.real(xuv_t)), np.max(np.real(xuv_t))], color='black')
        plt.plot([indexmin, indexmax], [threshold, threshold], color='red', linestyle='dashed')
        plt.text(0.1, 0.8, 'bad samples:{}'.format(bad_samples), transform=plt.gca().transAxes)
        plt.show()
        plt.pause(0.001)
        return bad_samples, False

    return bad_samples, True






def generate_samples(tf_graphs, n_samples, filename, streak_params, xuv_coefs, sess):
    print('creating file: ' + filename)

    # create hdf5 file
    with tables.open_file(filename, mode='w') as hd5file:
        # create array for trace
        hd5file.create_earray(hd5file.root, 'trace', tables.Float64Atom(), shape=(0, len(streak_params["p_values"]) * len(streak_params["tau_values"])))

        # create array for XUV
        hd5file.create_earray(hd5file.root, 'xuv_coefs', tables.Float64Atom(), shape=(0, xuv_coefs))

        # create array for IR
        hd5file.create_earray(hd5file.root, 'ir_params', tables.Float64Atom(), shape=(0, 4))

        hd5file.close()


    # make a sample with no phase to give a comparison
    xuv_coefs_in = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
    xuv_t = sess.run(tf_graphs["xuv_E_prop"]["t"], feed_dict={tf_graphs["xuv_coefs_in"]: xuv_coefs_in})
    threshold = np.max(np.abs(xuv_t)) / 200

    # count the number of bad samples generated
    bad_samples = 0


    # open and append the file
    with tables.open_file(filename, mode='a') as hd5file:
        for i in range(n_samples):

            xuv_good = False
            indexmin = 100
            indexmax = 924
            while not xuv_good:

                # xuv_coefs_in = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
                xuv_coefs_in = (np.random.rand(5)-0.5).reshape(1, -1)

                # generate time pulse from these coefficients
                xuv_t = sess.run(tf_graphs["xuv_E_prop"]["t"], feed_dict={tf_graphs["xuv_coefs_in"]: xuv_coefs_in})

                # check if the pulse is constrained in time window
                bad_samples, xuv_good = check_time_boundary(indexmin, indexmax, threshold, xuv_t[0], bad_samples)

            print("xuv was good")

            xuv_t = sess.run(tf_graphs["xuv_E_prop"]["t"], feed_dict={tf_graphs["xuv_coefs_in"]: xuv_coefs_in})
            plt.figure(111)
            plt.plot(xuv_spectrum.spectrum.tmat, np.real(xuv_t[0]))
            plt.show()

            exit(0)











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
    xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_in, amplitude=9.0)

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

    with tf.Session() as sess:


        generate_samples(tf_graphs=tf_graphs, n_samples=1000,
                         filename="train3.hdf5", streak_params=streak_params,
                         xuv_coefs=xuv_phase_coeffs, sess=sess)





    with tf.Session() as sess:
        # between -0.5 and 0.5
        xuv_coefs = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        # between -1 and 1
        ir_values = np.array([[0.0, 0.0, 0.0, 0.0]])
        image_sample = sess.run(image, feed_dict={xuv_coefs_in: xuv_coefs,
                                               ir_values_in: ir_values})

        image_sample = add_shot_noise(image_sample)




