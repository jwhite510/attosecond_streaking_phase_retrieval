import numpy as np
import matplotlib.pyplot as plt
import tf_functions
import tensorflow as tf
import generate_data3
import supervised_retrieval



def create_sample(modelname, count_num):

    # open the file with retrieved coefficients
    with open(modelname+"_noise_test_measured.p", "rb") as file:
        retrieval_data_measured_trace = pickle.load(file)

    trace = retrieval_data_measured_trace["reconstructed_trace"]
    xuv_coefs = retrieval_data_measured_trace["retrieved_xuv_coefs"]

    noise_trace = generate_data3.add_shot_noise(trace, count_num)

    return xuv_coefs_in, noise_trace




if __name__ == "__main__":

    modelname = "MLMN_noise_resistant_net_angle_1"

    xuv_coefs, trace_recons = create_sample(modelname, 200)

    supervised_retrieval_1 = supervised_retrieval.SupervisedRetrieval(modelname)

    retrieved_xuv_coefs_2nd = supervised_retrieval_1.retrieve(trace_recons)["xuv_params_out"]
    xuv_coefs_in = tf.placeholder(tf.float32, shape=[None, phase_parameters.params.xuv_phase_coefs])
    generated_xuv = tf_functions.xuv_taylor_to_E(xuv_coefs_in)

    with tf.Session() as sess:

        xuv_actual = sess.run(generated_xuv, feed_dict={xuv_coefs_in:retrieved_xuv_coefs_2nd})


    # plot this
    # ax.plot(spectrum.tmat_as, np.abs(xuv_actual["t"][0])**2, color="black")






