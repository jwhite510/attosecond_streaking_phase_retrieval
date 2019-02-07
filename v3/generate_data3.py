import tables
import tensorflow as tf
import tf_functions
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt



def generate_samples(tf_graphs, n_samples, filename):
    print('creating file: ' + filename)

    # create hdf5 file
    with tables.open_file(filename, mode='w') as hd5file:
        # create array for trace
        hd5file.create_earray(hd5file.root, 'trace', tables.Float64Atom(), shape=(0, len(p_values) * len(tau_values)))

        # create array for XUV
        hd5file.create_earray(hd5file.root, 'xuv_coefs', tables.Float64Atom(), shape=(0, len(p_values) * len(tau_values)))

        # create array for IR
        hd5file.create_earray(hd5file.root, 'ir_params', tables.Float64Atom(), shape=(0, len(p_values) * len(tau_values)))


        hd5file.close()










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
    xuv_coefs_in = tf.placeholder(tf.float32, shape=[None, 5])
    xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_in, amplitude=10.0)

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
    image = tf_functions.streaking_trace(xuv_cropped_f_in=xuv_E_prop["f_cropped"][0],
                                         ir_cropped_f_in=ir_E_prop["f_cropped"][0],Ip=Ip)


    tf_graphs = {}
    tf_graphs["image"] = image
    tf_graphs["xuv_coefs_in"] = xuv_coefs_in
    tf_graphs["ir_values_in"] = ir_values_in

    generate_samples(tf_graphs=tf_graphs, n_samples=1000, filename="train3.hdf5")





    with tf.Session() as sess:
        # between -0.5 and 0.5
        xuv_coefs = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        # between -1 and 1
        ir_values = np.array([[0.0, 0.0, 0.0, 0.0]])
        image_sample = sess.run(image, feed_dict={xuv_coefs_in: xuv_coefs,
                                               ir_values_in: ir_values})

        image_sample = add_shot_noise(image_sample)




