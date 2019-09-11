import numpy as np
from xuv_spectrum import spectrum
import matplotlib.pyplot as plt
import tf_functions
import tensorflow as tf
import generate_data3
import supervised_retrieval
import pickle
from phase_parameters import params



def add_noise_to_reconstructed_trace(modelname, count_num):

    # open the file with retrieved coefficients
    with open(modelname+"_noise_test_measured.p", "rb") as file:
        retrieval_data_measured_trace = pickle.load(file)

    trace_recons = retrieval_data_measured_trace["reconstructed_trace"]
    xuv_coefs = retrieval_data_measured_trace["retrieved_xuv_coefs"]
    trace_meas = retrieval_data_measured_trace["measured_trace"]

    noise_trace_recons_added_noise = generate_data3.add_shot_noise(trace_recons, count_num)

    return xuv_coefs, noise_trace_recons_added_noise, trace_meas




if __name__ == "__main__":

    modelname = "MLMN_noise_resistant_net_angle_1"
    count_num = 200

    retrieved_xuv_coefs_1st, noise_trace_recons_added_noise, trace_meas = add_noise_to_reconstructed_trace(modelname, count_num)

    supervised_retrieval_1 = supervised_retrieval.SupervisedRetrieval(modelname)

    retrieved_xuv_coefs_2nd = supervised_retrieval_1.retrieve(noise_trace_recons_added_noise)["xuv_retrieved"]

    # define tensorflow graph
    xuv_coefs_in = tf.placeholder(tf.float32, shape=[None, params.xuv_phase_coefs])
    generated_xuv = tf_functions.xuv_taylor_to_E(xuv_coefs_in)

    with tf.Session() as sess:

        gen_xuv_out_1st = sess.run(generated_xuv, feed_dict={xuv_coefs_in:retrieved_xuv_coefs_1st})
        gen_xuv_out_2nd = sess.run(generated_xuv, feed_dict={xuv_coefs_in:retrieved_xuv_coefs_2nd})


    fig = plt.figure(figsize=(15, 5))
    # fig.subplots_adjust(left=0.05, right=0.95, hspace=0.5)
    gs = fig.add_gridspec(2,2)

    # the original measured trace
    ax = fig.add_subplot(gs[0,0])
    ax.pcolormesh(params.delay_values_fs, params.K, trace_meas, cmap="jet")
    ax.set_title("Experimentally Measured Trace")

    # retrieved from original measured
    ax = fig.add_subplot(gs[1,0])
    ax.plot(spectrum.tmat_as, np.abs(gen_xuv_out_1st["t"][0])**2, color="black")
    ax.set_title("Retrieved I(t)")

    # -------- reconstructed -----

    # reconstructed trace with added noise
    ax = fig.add_subplot(gs[0,1])
    ax.pcolormesh(params.delay_values_fs, params.K, noise_trace_recons_added_noise, cmap="jet")
    ax.set_title("Reconstructed Trace then add noise {} counts".format(count_num))

    # retrieve I(t) from reconstructed trace
    ax = fig.add_subplot(gs[1,1])
    ax.plot(spectrum.tmat_as, np.abs(gen_xuv_out_2nd["t"][0])**2, color="black")
    ax.set_title("Retrieved I(t)")

    plt.savefig("./retrieve_reconstructed_{}_{}".format(count_num, modelname)+"_4")


    # plot this
    # ax.plot(spectrum.tmat_as, np.abs(xuv_actual["t"][0])**2, color="black")

