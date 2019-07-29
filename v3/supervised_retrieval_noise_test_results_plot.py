import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tf_functions
import phase_parameters




if __name__ == "__main__":

    with open("./supervised_retrieval_noise_test.p", "rb") as file:
        obj = pickle.load(file)

    print("type(obj['retrieved'])", type(obj['retrieved']))
    xuv_coefs_in = tf.placeholder(tf.float32, shape=[None, phase_parameters.params.xuv_phase_coefs])
    acutal_xuv = tf_functions.xuv_taylor_to_E(xuv_coefs_in)
    with tf.Session() as sess:

        for measured_trace, retrieved, count_num, xuv_input_coefs in zip(obj["measured_trace"], obj["retrieved"], obj["count_num"], obj["xuv_input_coefs"]):

            xuv_actual = sess.run(acutal_xuv, feed_dict={xuv_coefs_in:xuv_input_coefs})

            # measured trace
            measured_trace
            # retrieved E
            retrieved
            # actual E
            xuv_actual
            # count number
            count_num

            # just add plotting

