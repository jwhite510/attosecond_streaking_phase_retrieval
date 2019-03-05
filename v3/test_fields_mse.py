import tensorflow as tf
import tf_functions
import numpy as np
import scipy.constants as sc
import tables
import shutil
import matplotlib.pyplot as plt
import os
import phase_parameters.params
import network3


tf_generator_graphs, streak_params = network3.initialize_xuv_ir_trace_graphs()



# define phase net
x_in = tf.placeholder(shape=(None, int(len(streak_params["p_values"]) * len(streak_params["tau_values"]))),
                      dtype=tf.float32)
phase_net_output, hold_prob = network3.phase_retrieval_net(input=x_in, streak_params=streak_params)


# create label placeholder for supervised learning
xuv_phase_coefs = phase_parameters.params.xuv_phase_coefs
total_coefs_params_length = int(xuv_phase_coefs + 4)
actual_coefs_params = tf.placeholder(tf.float32, shape=[None, total_coefs_params_length])
supervised_label_fields = network3.create_fields_label_from_coefs_params(actual_coefs_params)


# get label
get_data = network3.GetData(batch_size=10)
batch_x, batch_y = get_data.next_batch()
trace = batch_x[0]
coefs_params = batch_y[0]


# define cost function
phase_network_loss = tf.losses.mean_squared_error(labels=supervised_label_fields["xuv_ir_field_label"],
                                                      predictions=phase_net_output["xuv_ir_field_label"])

#phase_network_loss = tf.losses.mean_squared_error(labels=supervised_label_fields["actual_coefs_params"],
#                                                      predictions=phase_net_output["predicted_coefficients_params"])
phase_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
phase_network_train = phase_optimizer.minimize(phase_network_loss)




init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # initial network output
    out = sess.run(phase_net_output["xuv_ir_field_label"], feed_dict={x_in: trace.reshape(1, -1)})
    plt.figure(1)
    plt.plot(out.reshape(-1))


    # label
    out = sess.run(supervised_label_fields["xuv_ir_field_label"], feed_dict={actual_coefs_params: coefs_params.reshape(1, -1)})
    plt.figure(2)
    plt.plot(out.reshape(-1))


    # train the network
    for _ in range(999):
        sess.run(phase_network_train, feed_dict={x_in: trace.reshape(1, -1),
                                                 actual_coefs_params: coefs_params.reshape(1, -1)})


    # new network output
    out = sess.run(phase_net_output["xuv_ir_field_label"], feed_dict={x_in: trace.reshape(1, -1)})
    plt.figure(3)
    plt.plot(out.reshape(-1))


plt.show()











































