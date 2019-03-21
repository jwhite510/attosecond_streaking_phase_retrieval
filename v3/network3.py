import tensorflow as tf
import tf_functions
import numpy as np
import scipy.constants as sc
import tables
import shutil
import matplotlib.pyplot as plt
import os
import phase_parameters.params





class GetData():
    def __init__(self, batch_size):

        self.batch_counter = 0
        self.batch_index = 0
        self.batch_size = batch_size
        self.train_filename = 'train3.hdf5'
        self.test_filename = 'test3.hdf5'

        hdf5_file = tables.open_file(self.train_filename, mode="r")
        attstraces = hdf5_file.root.trace[:, :]
        self.samples = np.shape(attstraces)[0]

        hdf5_file.close()

    def next_batch(self):

        # retrieve the next batch of data from the data source
        hdf5_file = tables.open_file(self.train_filename, mode="r")

        xuv_coefs = hdf5_file.root.xuv_coefs[self.batch_index:self.batch_index + self.batch_size, :]
        ir_params = hdf5_file.root.ir_params[self.batch_index:self.batch_index + self.batch_size, :]
        appended_label_batch = np.append(xuv_coefs, ir_params, 1)

        trace_batch = hdf5_file.root.noise_trace[self.batch_index:self.batch_index + self.batch_size, :]

        hdf5_file.close()

        self.batch_index += self.batch_size

        return  trace_batch, appended_label_batch


    def evaluate_on_test_data(self):

        # this is used to evaluate the mean squared error of the data after every epoch
        hdf5_file = tables.open_file(self.test_filename, mode="r")

        xuv_coefs = hdf5_file.root.xuv_coefs[:, :]
        ir_params = hdf5_file.root.ir_params[:, :]
        appended_label_batch = np.append(xuv_coefs, ir_params, 1)

        trace_batch = hdf5_file.root.noise_trace[:, :]

        hdf5_file.close()

        return trace_batch, appended_label_batch



    def evaluate_on_train_data(self, samples):

        # this is used to evaluate the mean squared error of the data after every epoch
        hdf5_file = tables.open_file(self.train_filename, mode="r")

        xuv_coefs = hdf5_file.root.xuv_coefs[:samples, :]
        ir_params = hdf5_file.root.ir_params[:samples, :]
        appended_label_batch = np.append(xuv_coefs, ir_params, 1)

        trace_batch = hdf5_file.root.noise_trace[:samples, :]

        hdf5_file.close()

        return trace_batch, appended_label_batch


def log_base(x, base, translate):
    return tf.log(x+translate) / tf.log(base)


def create_fields_label_from_coefs_params(actual_coefs_params):

    xuv_coefs_actual = actual_coefs_params[:, 0:phase_parameters.params.xuv_phase_coefs]
    ir_params_actual = actual_coefs_params[:, phase_parameters.params.xuv_phase_coefs:]
    xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_actual)
    ir_E_prop = tf_functions.ir_from_params(ir_params_actual)
    xuv_ir_field_label = concat_fields(xuv=xuv_E_prop["f_cropped"], ir=ir_E_prop["f_cropped"])

    fields = {}
    fields["xuv_E_prop"] = xuv_E_prop
    fields["ir_E_prop"] = ir_E_prop
    fields["xuv_ir_field_label"] = xuv_ir_field_label
    fields["actual_coefs_params"] = actual_coefs_params

    return fields


def concat_fields(xuv, ir):

    xuv_concat = tf.concat([tf.real(xuv), tf.imag(xuv)], axis=1)
    ir_concat = tf.concat([tf.real(ir), tf.imag(ir)], axis=1)
    both_fields_concat = tf.concat([xuv_concat, ir_concat], axis=1)

    return both_fields_concat


def test_generate_data(nn_nodes):

    # generate a bunch of samples and test threshold value
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        fig = plt.figure()
        gs = fig.add_gridspec(2, 1)
        ax3 = fig.add_subplot(gs[:, :])
        plt.ion()

        # define threshold
        xuv_coefs_in = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        xuv_t = sess.run(nn_nodes["gan"]["xuv_E_prop"]["t"],
                         feed_dict={nn_nodes["gan"]["gan_xuv_out_nolin"]: xuv_coefs_in})
        threshold = np.max(np.abs(xuv_t[0])) * phase_parameters.params.threshold_scaler

        for _ in range(999):
            gan_in = np.random.random(100).reshape(1, -1)

            out = sess.run(nn_nodes["gan"]["xuv_E_prop"]["t"],
                           feed_dict={nn_nodes["gan"]["gan_input"]: gan_in})

            indexmax = phase_parameters.params.threshold_max_index
            indexmin = phase_parameters.params.threshold_min_index

            indexmin_value = np.max(np.abs(out[0, :indexmin]))
            indexmax_value = np.max(np.abs(out[0, indexmax:]))

            ax3.cla()
            ax3.plot(np.real(out[0]), color="blue")
            ax3.plot(np.abs(out[0]), color="black")
            ax3.plot([indexmin, indexmin], [indexmin_value, 0], color="red")
            ax3.plot([indexmax, indexmax], [indexmax_value, 0], color="red")
            ax3.plot([indexmin, indexmax], [threshold, threshold], color="orange", linestyle="dashed")

            # print(indexmin_value)
            # print(indexmax_value)
            # print(threshold)

            if indexmin_value > threshold or indexmax_value > threshold:
                print("exceeded threshold")
                plt.ioff()
                plt.show()
                exit(0)

            plt.pause(0.001)


def separate_xuv_ir_vec(xuv_ir_vec):

    xuv = xuv_ir_vec[0:5]
    ir = xuv_ir_vec[5:]

    return xuv, ir


def plot_predictions(x_in, y_in, indexes, axes, figure, epoch, set, net_name, nn_nodes, sess, streak_params):

    # get find where in the vector is the ir and xuv

    print("plot predicitons")


    for j, index in enumerate(indexes):

        mse = sess.run(nn_nodes["supervised"]["phase_network_fields_loss"],
                       feed_dict={nn_nodes["general"]["x_in"]: x_in[index].reshape(1, -1),
                                  nn_nodes["supervised"]["actual_coefs_params"]: y_in[index].reshape(1, -1)})

        # get the actual fields
        actual_xuv_field = sess.run(nn_nodes["supervised"]["supervised_label_fields"]["xuv_E_prop"]["f_cropped"],
                       feed_dict={nn_nodes["supervised"]["actual_coefs_params"]: y_in[index].reshape(1, -1)})

        actual_ir_field = sess.run(nn_nodes["supervised"]["supervised_label_fields"]["ir_E_prop"]["f_cropped"],
                       feed_dict={nn_nodes["supervised"]["actual_coefs_params"]: y_in[index].reshape(1, -1)})

        # get the predicted fields
        predicted_xuv_field = sess.run(nn_nodes["general"]["phase_net_output"]["xuv_E_prop"]["f_cropped"],
                                       feed_dict={nn_nodes["general"]["x_in"]: x_in[index].reshape(1, -1)})

        predicted_ir_field = sess.run(nn_nodes["general"]["phase_net_output"]["ir_E_prop"]["f_cropped"],
                                       feed_dict={nn_nodes["general"]["x_in"]: x_in[index].reshape(1, -1)})

        actual_xuv_field = actual_xuv_field.reshape(-1)
        actual_ir_field = actual_ir_field.reshape(-1)
        predicted_xuv_field = predicted_xuv_field.reshape(-1)
        predicted_ir_field = predicted_ir_field.reshape(-1)

        # calculate generated streaking trace
        generated_trace = sess.run(nn_nodes["general"]["reconstructed_trace"],
                                   feed_dict={nn_nodes["general"]["x_in"]: x_in[index].reshape(1, -1)})


        axes[j]['input_trace'].cla()
        axes[j]['input_trace'].pcolormesh(x_in[index].reshape(len(streak_params["p_values"]), len(streak_params["tau_values"])), cmap='jet')
        axes[j]['input_trace'].text(0.0, 1.0, 'input_trace', transform=axes[j]['input_trace'].transAxes,backgroundcolor='white')
        axes[j]['input_trace'].set_xticks([])
        axes[j]['input_trace'].set_yticks([])

        axes[j]['actual_xuv'].cla()
        axes[j]['actual_xuv_twinx'].cla()
        axes[j]['actual_xuv'].plot(np.real(actual_xuv_field), color='blue', alpha=0.3)
        axes[j]['actual_xuv'].plot(np.imag(actual_xuv_field), color='red', alpha=0.3)
        axes[j]['actual_xuv'].plot(np.abs(actual_xuv_field), color='black')
        # plot the phase
        axes[j]['actual_xuv_twinx'].plot(np.unwrap(np.angle(actual_xuv_field)), color='green')
        axes[j]['actual_xuv_twinx'].tick_params(axis='y', colors='green')
        axes[j]['actual_xuv'].text(0.0,1.0, 'actual_xuv', transform=axes[j]['actual_xuv'].transAxes, backgroundcolor='white')
        axes[j]['actual_xuv'].set_xticks([])
        axes[j]['actual_xuv'].set_yticks([])

        axes[j]['predict_xuv'].cla()
        axes[j]['predict_xuv_twinx'].cla()
        axes[j]['predict_xuv'].plot(np.real(predicted_xuv_field), color='blue', alpha=0.3)
        axes[j]['predict_xuv'].plot(np.imag(predicted_xuv_field), color='red', alpha=0.3)
        axes[j]['predict_xuv'].plot(np.abs(predicted_xuv_field), color='black')
        #plot the phase
        axes[j]['predict_xuv_twinx'].plot(np.unwrap(np.angle(predicted_xuv_field)), color='green')
        axes[j]['predict_xuv_twinx'].tick_params(axis='y', colors='green')
        axes[j]['predict_xuv'].text(0.0, 1.0, 'predict_xuv', transform=axes[j]['predict_xuv'].transAxes, backgroundcolor='white')
        axes[j]['predict_xuv'].text(-0.4, 0, 'MSE: {} '.format(str(mse)),
                                   transform=axes[j]['predict_xuv'].transAxes, backgroundcolor='white')
        axes[j]['predict_xuv'].set_xticks([])
        axes[j]['predict_xuv'].set_yticks([])

        axes[j]['actual_ir'].cla()
        axes[j]['actual_ir'].plot(np.real(actual_ir_field), color='blue')
        axes[j]['actual_ir'].plot(np.imag(actual_ir_field), color='red')
        axes[j]['actual_ir'].text(0.0, 1.0, 'actual_ir', transform=axes[j]['actual_ir'].transAxes, backgroundcolor='white')

        if j == 0:

            axes[j]['actual_ir'].text(0.5, 1.25, net_name, transform=axes[j]['actual_ir'].transAxes,
                                      backgroundcolor='white')

            axes[j]['actual_ir'].text(0.5, 1.1, set, transform=axes[j]['actual_ir'].transAxes,
                                      backgroundcolor='white')

        axes[j]['actual_ir'].set_xticks([])
        axes[j]['actual_ir'].set_yticks([])

        axes[j]['predict_ir'].cla()
        axes[j]['predict_ir'].plot(np.real(predicted_ir_field), color='blue')
        axes[j]['predict_ir'].plot(np.imag(predicted_ir_field), color='red')
        axes[j]['predict_ir'].text(0.0, 1.0, 'predict_ir', transform=axes[j]['predict_ir'].transAxes,backgroundcolor='white')
        axes[j]['predict_ir'].set_xticks([])
        axes[j]['predict_ir'].set_yticks([])



        axes[j]['reconstruct'].pcolormesh(generated_trace,cmap='jet')
        axes[j]['reconstruct'].text(0.0, 1.0, 'reconstructed_trace', transform=axes[j]['reconstruct'].transAxes,backgroundcolor='white')
        axes[j]['reconstruct'].set_xticks([])
        axes[j]['reconstruct'].set_yticks([])




        # save image
        dir = "./nnpictures/" + modelname + "/" + set + "/"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        figure.savefig(dir + str(epoch) + ".png")


def update_plots(data_obj, sess, nn_nodes, modelname, epoch, axes, streak_params):

    batch_x_train, batch_y_train = data_obj.evaluate_on_train_data(samples=500)
    plot_predictions(x_in=batch_x_train, y_in=batch_y_train, indexes=[0, 1, 2],
                      axes=axes["trainplot1"], figure=axes["trainfig1"], epoch=epoch, set='train_data_1',
                      net_name=modelname, nn_nodes=nn_nodes, sess=sess,
                     streak_params=streak_params)

    plot_predictions(x_in=batch_x_train, y_in=batch_y_train, indexes=[3, 4, 5],
                     axes=axes["trainplot2"], figure=axes["trainfig2"], epoch=epoch, set='train_data_2',
                     net_name=modelname, nn_nodes=nn_nodes, sess=sess,
                     streak_params=streak_params)


    batch_x_test, batch_y_test = data_obj.evaluate_on_test_data()
    plot_predictions(x_in=batch_x_test, y_in=batch_y_test, indexes=[0, 1, 2],
                     axes=axes["testplot1"], figure=axes["testfig1"], epoch=epoch, set='test_data_1',
                     net_name=modelname, nn_nodes=nn_nodes, sess=sess,
                     streak_params=streak_params)

    plot_predictions(x_in=batch_x_test, y_in=batch_y_test, indexes=[3, 4, 5],
                     axes=axes["testplot2"], figure=axes["testfig2"], epoch=epoch, set='test_data_2',
                     net_name=modelname, nn_nodes=nn_nodes, sess=sess,
                     streak_params=streak_params)

    plt.show()
    plt.pause(0.001)


def init_tf_loggers(nn_nodes):

    test_mse_tb_phasecurve = tf.summary.scalar("test_mse_phasecurve", nn_nodes["supervised"]["phase_network_phasecurve_loss"])
    train_mse_tb_phasecurve = tf.summary.scalar("train_mse_phasecurve", nn_nodes["supervised"]["phase_network_phasecurve_loss"])

    test_mse_tb_fields = tf.summary.scalar("test_mse_fields", nn_nodes["supervised"]["phase_network_fields_loss"])
    train_mse_tb_fields = tf.summary.scalar("train_mse_fields", nn_nodes["supervised"]["phase_network_fields_loss"])

    test_mse_tb_coefs_params = tf.summary.scalar("test_mse_coef_params", nn_nodes["supervised"]["phase_network_coefs_params_loss"])
    train_mse_tb_coefs_params = tf.summary.scalar("train_mse_coef_params", nn_nodes["supervised"]["phase_network_coefs_params_loss"])

    tf_loggers = {}
    tf_loggers["test_mse_tb_phasecurve"] = test_mse_tb_phasecurve
    tf_loggers["train_mse_tb_phasecurve"] = train_mse_tb_phasecurve
    tf_loggers["test_mse_tb_fields"] = test_mse_tb_fields
    tf_loggers["train_mse_tb_fields"] = train_mse_tb_fields
    tf_loggers["test_mse_tb_coefs_params"] = test_mse_tb_coefs_params
    tf_loggers["train_mse_tb_coefs_params"] = train_mse_tb_coefs_params

    return tf_loggers


def add_tensorboard_values(nn_nodes, tf_loggers):


    #***********************************
    # ..................................
    # ..........test set...............
    # ..................................
    #***********************************
    # view the mean squared error of the test data
    batch_x_test, batch_y_test = get_data.evaluate_on_test_data()


    #---------------------------------
    # -------phase curve loss---------
    #---------------------------------
    print("Phasecurve test MSE: ", sess.run(nn_nodes["supervised"]["phase_network_phasecurve_loss"],
                                        feed_dict={nn_nodes["supervised"]["x_in"]: batch_x_test,
                                                   nn_nodes["supervised"]["actual_coefs_params"]: batch_y_test}))
    summ = sess.run(tf_loggers["test_mse_tb_phasecurve"],
                    feed_dict={nn_nodes["supervised"]["x_in"]: batch_x_test,
                               nn_nodes["supervised"]["actual_coefs_params"]: batch_y_test})
    writer.add_summary(summ, global_step=i + 1)

    # ---------------------------------
    # ----------fields loss------------
    # ---------------------------------
    print("fields test MSE: ", sess.run(nn_nodes["supervised"]["phase_network_fields_loss"],
                                 feed_dict={nn_nodes["supervised"]["x_in"]: batch_x_test,
                                            nn_nodes["supervised"]["actual_coefs_params"]: batch_y_test}))
    summ = sess.run(tf_loggers["test_mse_tb_fields"],
                    feed_dict={nn_nodes["supervised"]["x_in"]: batch_x_test,
                               nn_nodes["supervised"]["actual_coefs_params"]: batch_y_test})
    writer.add_summary(summ, global_step=i + 1)

    # ---------------------------------
    # --------coef params loss---------
    # ---------------------------------
    print("coefs params test MSE: ", sess.run(nn_nodes["supervised"]["phase_network_coefs_params_loss"],
                                 feed_dict={nn_nodes["supervised"]["x_in"]: batch_x_test,
                                            nn_nodes["supervised"]["actual_coefs_params"]: batch_y_test}))
    summ = sess.run(tf_loggers["test_mse_tb_coefs_params"],
                    feed_dict={nn_nodes["supervised"]["x_in"]: batch_x_test,
                               nn_nodes["supervised"]["actual_coefs_params"]: batch_y_test})
    writer.add_summary(summ, global_step=i + 1)


    #***********************************
    # ..................................
    # ..........train set................
    # ..................................
    #***********************************
    # view the mean squared error of the train data
    batch_x_train, batch_y_train = get_data.evaluate_on_train_data(samples=500)

    # ----------------------------------------
    # ----------phase curve loss--------------
    # ----------------------------------------
    print("Phasecurve train MSE: ", sess.run(nn_nodes["supervised"]["phase_network_phasecurve_loss"],
                                         feed_dict={nn_nodes["supervised"]["x_in"]: batch_x_train,
                                                    nn_nodes["supervised"]["actual_coefs_params"]: batch_y_train}))
    summ = sess.run(tf_loggers["train_mse_tb_phasecurve"],
                    feed_dict={nn_nodes["supervised"]["x_in"]: batch_x_train,
                               nn_nodes["supervised"]["actual_coefs_params"]: batch_y_train})
    writer.add_summary(summ, global_step=i + 1)


    #----------------------------------------
    # ----------fields loss------------------
    #----------------------------------------
    print("fields train MSE: ", sess.run(nn_nodes["supervised"]["phase_network_fields_loss"],
                                        feed_dict={nn_nodes["supervised"]["x_in"]: batch_x_train,
                                                   nn_nodes["supervised"]["actual_coefs_params"]: batch_y_train}))
    summ = sess.run(tf_loggers["train_mse_tb_fields"],
                    feed_dict={nn_nodes["supervised"]["x_in"]: batch_x_train,
                               nn_nodes["supervised"]["actual_coefs_params"]: batch_y_train})
    writer.add_summary(summ, global_step=i + 1)

    #-----------------------------------------
    # -----------coef params loss--------------
    #-----------------------------------------
    print("coefs params train MSE: ", sess.run(nn_nodes["supervised"]["phase_network_coefs_params_loss"],
                                              feed_dict={nn_nodes["supervised"]["x_in"]: batch_x_train,
                                                         nn_nodes["supervised"]["actual_coefs_params"]: batch_y_train}))
    summ = sess.run(tf_loggers["train_mse_tb_coefs_params"],
                    feed_dict={nn_nodes["supervised"]["x_in"]: batch_x_train,
                               nn_nodes["supervised"]["actual_coefs_params"]: batch_y_train})
    writer.add_summary(summ, global_step=i + 1)

    # ..................................
    # .....write to tensorboard.........
    # ..................................
    writer.flush()


def show_loading_bar(dots):
    # display loading bar
    percent = 50 * get_data.batch_index / get_data.samples
    if percent - dots > 1:
        print(".", end="", flush=True)
        dots += 1
    return dots


def create_sample_plot(samples_per_plot=3):

    fig = plt.figure(figsize=(16, 8))
    plt.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.05,
                            wspace=0.2, hspace=0.1)
    gs = fig.add_gridspec(4,int(samples_per_plot*2))

    plot_rows = []
    for i in range(samples_per_plot):

        column_axes = {}

        column_axes['actual_ir'] = fig.add_subplot(gs[0, 2*i])
        column_axes['actual_xuv'] = fig.add_subplot(gs[0, 2*i+1])
        column_axes['actual_xuv_twinx'] = column_axes['actual_xuv'].twinx()

        column_axes['input_trace'] = fig.add_subplot(gs[1, 2*i:2*i+2])

        column_axes['predict_ir'] = fig.add_subplot(gs[2, 2*i])
        column_axes['predict_xuv'] = fig.add_subplot(gs[2, 2*i+1])
        column_axes['predict_xuv_twinx'] = column_axes['predict_xuv'].twinx()

        column_axes['reconstruct'] = fig.add_subplot(gs[3, 2*i:2*i+2])

        plot_rows.append(column_axes)

    return plot_rows, fig


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding='SAME')


def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(init_bias_vals)


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


def multires_layer(input, input_channels, filter_sizes, stride=1):

    # list of layers
    filters = []
    for filter_size in filter_sizes:
        # create filter
        filters.append(convolutional_layer(input, shape=[filter_size, filter_size,
                        input_channels, input_channels], activate='relu', stride=[stride, stride]))

    concat_layer = tf.concat(filters, axis=3)
    return concat_layer


def convolutional_layer(input_x, shape, activate, stride):
    W = init_weights(shape)
    b = init_bias([shape[3]])

    if activate == 'relu':
        return tf.nn.relu(conv2d(input_x, W, stride) + b)

    if activate == 'leaky':
        return tf.nn.leaky_relu(conv2d(input_x, W, stride) + b)

    elif activate == 'none':
        return conv2d(input_x, W, stride) + b


def initialize_xuv_ir_trace_graphs():

    # initialize XUV generator
    xuv_phase_coeffs = phase_parameters.params.xuv_phase_coefs
    xuv_coefs_in = tf.placeholder(tf.float32, shape=[None, xuv_phase_coeffs])
    xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_in)


    # IR creation
    ir_values_in = tf.placeholder(tf.float32, shape=[None, 4])
    ir_E_prop = tf_functions.ir_from_params(ir_values_in)

    # initialize streaking trace generator
    # Neon


    # construct streaking image
    image, streak_params = tf_functions.streaking_trace(xuv_cropped_f_in=xuv_E_prop["f_cropped"][0],
                                                        ir_cropped_f_in=ir_E_prop["f_cropped"][0],
                                                        )

    tf_graphs = {}
    tf_graphs["xuv_coefs_in"] = xuv_coefs_in
    tf_graphs["ir_values_in"] = ir_values_in
    tf_graphs["xuv_E_prop"] = xuv_E_prop
    tf_graphs["ir_E_prop"] = ir_E_prop
    tf_graphs["image"] = image

    return tf_graphs, streak_params


def gan_network(input):

    xuv_phase_coefs = phase_parameters.params.xuv_phase_coefs
    output_length = xuv_phase_coefs - 1 + 4     # remove 1 for no linear phase...
                                                # add 4 for ir params


    with tf.variable_scope("gan"):
        hidden1 = tf.layers.dense(inputs=input, units=128)

        alpha = 0.01
        hidden1 = tf.maximum(alpha * hidden1, hidden1)

        hidden2 = tf.layers.dense(inputs=hidden1, units=128)

        hidden2 = tf.maximum(alpha * hidden2, hidden2)

        # output of neural net between -1 and 1
        output = tf.layers.dense(hidden2, units=output_length, activation=tf.nn.tanh)

        # scaling factor between 0 and 1
        s_out = tf.layers.dense(hidden2, units=1, activation=tf.nn.sigmoid)

        # output : [---xuv_coefs-- --ir_params--]
        # represent taylor series coefficients
        xuv_out = output[:, 0:xuv_phase_coefs - 1]
        ir_out = output[:, xuv_phase_coefs - 1:]

        # append 0 to the xuv output because 0 linear phase
        samples_in = tf.shape(xuv_out)[0]
        zeros_vec = tf.fill([samples_in, 1], 0.0)
        xuv_out_nolin = tf.concat([zeros_vec, xuv_out], axis=1)

        # normalize the xuv output
        summation = tf.reduce_sum(tf.abs(xuv_out_nolin), axis=1)
        quotient = (1 / s_out) * summation
        xuv_coefs_normalized = xuv_out_nolin / quotient

        # create label with coefs and params
        coefs_params_label = tf.concat([xuv_coefs_normalized, ir_out], axis=1)

        # generate complex fields from these coefs
        xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_normalized)
        ir_E_prop = tf_functions.ir_from_params(ir_out)

        # concat these vectors to make a label
        xuv_ir_field_label = concat_fields(xuv=xuv_E_prop["f_cropped"], ir=ir_E_prop["f_cropped"])

        outputs = {}
        outputs["xuv_ir_field_label"] = xuv_ir_field_label
        outputs["coefs_params_label"] = coefs_params_label
        outputs["ir_E_prop"] = ir_E_prop
        outputs["xuv_E_prop"] = xuv_E_prop


        return outputs


def phase_retrieval_net(input, streak_params):

    xuv_phase_coefs = phase_parameters.params.xuv_phase_coefs
    total_coefs_params_length = int(xuv_phase_coefs + 4)

    # define phase retrieval neural network
    with tf.variable_scope("phase"):
        # input image
        x_image = tf.reshape(input, [-1, len(streak_params["p_values"]), len(streak_params["tau_values"]), 1])

        # six convolutional layers
        multires_filters = [11, 7, 5, 3]

        multires_layer_1 = multires_layer(input=x_image, input_channels=1, filter_sizes=multires_filters)

        conv_layer_1 = convolutional_layer(multires_layer_1,
                                           shape=[1, 1, len(multires_filters), 2 * len(multires_filters)],
                                           activate='relu', stride=[2, 2])

        multires_layer_2 = multires_layer(input=conv_layer_1, input_channels=2 * len(multires_filters),
                                          filter_sizes=multires_filters)

        conv_layer_2 = convolutional_layer(multires_layer_2,
                                           shape=[1, 1, 32, 64], activate='relu', stride=[2, 2])

        multires_layer_3 = multires_layer(input=conv_layer_2, input_channels=64,
                                          filter_sizes=multires_filters)

        conv_layer_3 = convolutional_layer(multires_layer_3,
                                           shape=[1, 1, 256,
                                                  512], activate='relu', stride=[2, 2])

        convo_3_flat = tf.contrib.layers.flatten(conv_layer_3)
        full_layer_one = normal_full_layer(convo_3_flat, 1024)
        #full_layer_one = normal_full_layer(convo_3_flat, 2)
        #print("layer needs to be set to 1024!!")

        # dropout
        hold_prob = tf.placeholder_with_default(1.0, shape=())
        dropout_layer = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

        # neural net output coefficients
        predicted_coefficients_params = normal_full_layer(dropout_layer, total_coefs_params_length)
        # predicted_coefficients_params = tf.nn.tanh(normal_full_layer(dropout_layer, total_coefs_params_length))

        xuv_coefs_pred = predicted_coefficients_params[:, 0:phase_parameters.params.xuv_phase_coefs]
        ir_params_pred = predicted_coefficients_params[:, phase_parameters.params.xuv_phase_coefs:]

        # generate fields from coefficients
        xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_pred)
        ir_E_prop = tf_functions.ir_from_params(ir_params_pred)

        # generate a label from the complex fields
        xuv_ir_field_label = concat_fields(xuv=xuv_E_prop["f_cropped"], ir=ir_E_prop["f_cropped"])


        phase_net_output = {}
        phase_net_output["xuv_ir_field_label"] = xuv_ir_field_label
        phase_net_output["ir_E_prop"] = ir_E_prop
        phase_net_output["xuv_E_prop"] = xuv_E_prop
        phase_net_output["predicted_coefficients_params"] = predicted_coefficients_params

        return phase_net_output, hold_prob


def setup_neural_net(streak_params):

    xuv_phase_coefs = phase_parameters.params.xuv_phase_coefs

    print('Setting up multires layer network with more conv weights')

    # define the label for supervised learning of phase retrieval net
    total_coefs_params_length = int(xuv_phase_coefs + 4)

    # define GAN network
    gan_input = tf.placeholder(tf.float32, shape=[1, 100])

    # GAN output is used to create XUV field and streaking trace
    gan_output = gan_network(input=gan_input)

    # use the fields to generate streaking trace
    # sample size of one required as of now
    x, _ = tf_functions.streaking_trace(xuv_cropped_f_in=gan_output["xuv_E_prop"]["f_cropped"][0],
                                            ir_cropped_f_in=gan_output["ir_E_prop"]["f_cropped"][0])
    x_flat = tf.reshape(x, [1, -1])
    # this placeholder accepts either an input as placeholder (supervised learning)
    # or it will default to the GAN generated fields as input
    x_in = tf.placeholder_with_default(x_flat, shape=(None, int(len(streak_params["p_values"]) * len(streak_params["tau_values"]))))


    # pass image through phase retrieval network
    phase_net_output, hold_prob = phase_retrieval_net(input=x_in, streak_params=streak_params)


    # create label for supervised learning
    actual_coefs_params = tf.placeholder(tf.float32, shape=[None, total_coefs_params_length])
    supervised_label_fields = create_fields_label_from_coefs_params(actual_coefs_params)


    # generate the reconstructed trace
    reconstructed_trace, _ = tf_functions.streaking_trace(xuv_cropped_f_in=phase_net_output["xuv_E_prop"]["f_cropped"][0],
                                                          ir_cropped_f_in=phase_net_output["ir_E_prop"]["f_cropped"][0])

    # generate proof trace
    reconstructed_proof = tf_functions.proof_trace(reconstructed_trace)


    # input proof trace
    x_in_reshaped = tf.reshape(x_in, [len(streak_params["p_values"]), len(streak_params["tau_values"])])
    input_image_proof = tf_functions.proof_trace(x_in_reshaped)


    # divide the variables to train with gan and phase retrieval net individually
    tvars = tf.trainable_variables()
    phase_net_vars = [var for var in tvars if "phase" in var.name]
    gan_net_vars = [var for var in tvars if "gan" in var.name]


    #........................................................
    #........................................................
    # ..............define loss functions....................
    #........................................................
    #........................................................




    #........................................................
    # .............GAN NETWORK LOSS FUNCTIONS................
    #........................................................
    # maximize loss between complex fields
    gan_fields_loss = (1/tf.losses.mean_squared_error(labels=gan_output["xuv_ir_field_label"],
                                                       predictions=phase_net_output["xuv_ir_field_label"]))
    gan_LR = tf.placeholder(tf.float32, shape=[])
    gan_optimizer = tf.train.AdamOptimizer(learning_rate=gan_LR)
    gan_network_train = gan_optimizer.minimize(gan_fields_loss, var_list=gan_net_vars)




    # ........................................................
    # ........SUPERVISED LEARNING LOSS FUNCTIONS..............
    # ........................................................
    s_LR = tf.placeholder(tf.float32, shape=[])

    # phase curve loss function
    phase_network_phasecurve_loss = tf.losses.mean_squared_error(labels=supervised_label_fields["xuv_E_prop"]["phasecurve_cropped"],
                                                        predictions=phase_net_output["xuv_E_prop"]["phasecurve_cropped"])
    phase_phasecurve_optimizer = tf.train.AdamOptimizer(learning_rate=s_LR)
    phase_network_train_phasecurve = phase_phasecurve_optimizer.minimize(phase_network_phasecurve_loss, var_list=phase_net_vars)

    # fields loss function for training phase retrieval network
    phase_network_fields_loss = tf.losses.mean_squared_error(labels=supervised_label_fields["xuv_ir_field_label"],
                                                        predictions=phase_net_output["xuv_ir_field_label"])
    phase_fields_optimizer = tf.train.AdamOptimizer(learning_rate=s_LR)
    phase_network_train_fields = phase_fields_optimizer.minimize(phase_network_fields_loss, var_list=phase_net_vars)

    # coefs and params loss function for training phase retrieval network
    phase_network_coefs_params_loss = tf.losses.mean_squared_error(labels=supervised_label_fields["actual_coefs_params"],
                                                        predictions=phase_net_output["predicted_coefficients_params"])
    phase_coefs_params_optimizer = tf.train.AdamOptimizer(learning_rate=s_LR)
    phase_network_train_coefs_params = phase_coefs_params_optimizer.minimize(phase_network_coefs_params_loss, var_list=phase_net_vars)




    # ..........................................................
    # .........UNSUPERVISED LEARNING LOSS FUNCTION..............
    # ..........................................................
    u_LR = tf.placeholder(tf.float32, shape=[])

    # regular cost function
    unsupervised_learning_loss = tf.losses.mean_squared_error(labels=x_in,
                                                              predictions=tf.reshape(reconstructed_trace, [1, -1]))
    unsupervised_optimizer = tf.train.AdamOptimizer(learning_rate=u_LR)
    unsupervised_train = unsupervised_optimizer.minimize(unsupervised_learning_loss,
                                                        var_list=phase_net_vars)

    # log cost function
    # log1 = log_base(x=0.5, base=10.0, translate=1)
    u_base = tf.placeholder(tf.float32, shape=[])
    u_translate = tf.placeholder(tf.float32, shape=[])
    unsupervised_learning_loss_log = tf.losses.mean_squared_error(
                            labels=log_base(x=x_in, base=u_base, translate=u_translate),
                            predictions=log_base(x=tf.reshape(reconstructed_trace, [1, -1]),
                                                 base=u_base,
                                                 translate=u_translate)
    )
    unsupervised_optimizer_log = tf.train.AdamOptimizer(learning_rate=u_LR)
    unsupervised_train_log = unsupervised_optimizer_log.minimize(unsupervised_learning_loss_log,
                                                        var_list=phase_net_vars)



    # ..........................................................
    # .................PROOF RETRIEVAL LOSS FUNC................
    # ..........................................................
    # regular cost function
    proof_unsupervised_learning_loss = tf.losses.mean_squared_error(labels=tf.reshape(input_image_proof["proof"], [1, -1]),
                                                              predictions=tf.reshape(reconstructed_proof["proof"], [1, -1]))
    proof_unsupervised_optimizer = tf.train.AdamOptimizer(learning_rate=u_LR)
    proof_unsupervised_train = proof_unsupervised_optimizer.minimize(proof_unsupervised_learning_loss,
                                                         var_list=phase_net_vars)




    # ..........................................................
    # ...................DEFINE NODES FOR USE...................
    # ..........................................................

    nn_nodes = {}
    nn_nodes["gan"] = {}
    nn_nodes["supervised"] = {}
    nn_nodes["unsupervised"] = {}
    nn_nodes["general"] = {}

    nn_nodes["gan"]["gan_input"] = gan_input
    nn_nodes["gan"]["gan_output"] = gan_output
    nn_nodes["gan"]["gan_LR"] = gan_LR
    nn_nodes["gan"]["gan_network_train"] = gan_network_train

    nn_nodes["supervised"]["x_in"] = x_in
    nn_nodes["supervised"]["actual_coefs_params"] = actual_coefs_params
    nn_nodes["supervised"]["phase_network_train_phasecurve"] = phase_network_train_phasecurve
    nn_nodes["supervised"]["phase_network_train_fields"] = phase_network_train_fields
    nn_nodes["supervised"]["phase_network_train_coefs_params"] = phase_network_train_coefs_params
    nn_nodes["supervised"]["s_LR"] = s_LR
    nn_nodes["supervised"]["phase_network_phasecurve_loss"] = phase_network_phasecurve_loss
    nn_nodes["supervised"]["phase_network_fields_loss"] = phase_network_fields_loss
    nn_nodes["supervised"]["phase_network_coefs_params_loss"] = phase_network_coefs_params_loss
    nn_nodes["supervised"]["supervised_label_fields"] = supervised_label_fields


    nn_nodes["unsupervised"]["x_in"] = x_in
    nn_nodes["unsupervised"]["unsupervised_train"] = unsupervised_train
    nn_nodes["unsupervised"]["unsupervised_train_log"] = unsupervised_train_log
    nn_nodes["unsupervised"]["u_LR"] = u_LR
    nn_nodes["unsupervised"]["unsupervised_learning_loss"] = unsupervised_learning_loss
    nn_nodes["unsupervised"]["unsupervised_learning_loss_log"] = unsupervised_learning_loss_log
    nn_nodes["unsupervised"]["u_base"] = u_base
    nn_nodes["unsupervised"]["u_translate"] = u_translate

    nn_nodes["unsupervised"]["proof"] = {}
    nn_nodes["unsupervised"]["proof"]["x_in"] = x_in
    nn_nodes["unsupervised"]["proof"]["u_LR"] = u_LR
    nn_nodes["unsupervised"]["proof"]["reconstructed_proof"] = reconstructed_proof
    nn_nodes["unsupervised"]["proof"]["input_image_proof"] = input_image_proof
    nn_nodes["unsupervised"]["proof"]["proof_unsupervised_train"] = proof_unsupervised_train
    nn_nodes["unsupervised"]["proof"]["proof_unsupervised_learning_loss"] = proof_unsupervised_learning_loss




    nn_nodes["general"]["phase_net_output"] = phase_net_output
    nn_nodes["general"]["reconstructed_trace"] = reconstructed_trace
    nn_nodes["general"]["hold_prob"] = hold_prob
    nn_nodes["general"]["x_in"] = x_in

    return nn_nodes



if __name__ == "__main__":



    # initialize xuv, IR, and trace graphs
    tf_generator_graphs, streak_params = initialize_xuv_ir_trace_graphs()

    # build neural net graph
    nn_nodes = setup_neural_net(streak_params)

    # test_generate_data(nn_nodes)

    print("built neural net")

    # init data object
    get_data = GetData(batch_size=10)

    # initialize mse tracking objects
    tf_loggers = init_tf_loggers(nn_nodes)


    # saver and set epoch number to run
    saver = tf.train.Saver()
    epochs = 900000

    # set the name of the neural net test run and save the settigns
    modelname = 'test1_phasecurve_proof'

    print('starting ' + modelname)

    shutil.copyfile('./network3.py', './models/network3_{}.py'.format(modelname))

    # create figures for showing results
    axes = {}

    axes["testplot1"], axes["testfig1"]= create_sample_plot()
    axes["testplot2"], axes["testfig2"]= create_sample_plot()

    axes["trainplot1"], axes["trainfig1"]= create_sample_plot()
    axes["trainplot2"], axes["trainfig2"]= create_sample_plot()

    plt.ion()

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        writer = tf.summary.FileWriter("./tensorboard_graph/" + modelname)

        for i in range(epochs):
            print("Epoch : {}".format(i + 1))

            # iterate through every sample in the training set
            dots = 0
            while get_data.batch_index < get_data.samples:

                dots = show_loading_bar(dots)

                # retrieve data
                batch_x, batch_y = get_data.next_batch()

                # train network
                if i < 15:
                    sess.run(nn_nodes["supervised"]["phase_network_train_coefs_params"],
                             feed_dict={nn_nodes["supervised"]["x_in"]: batch_x,
                                        nn_nodes["supervised"]["actual_coefs_params"]: batch_y,
                                        nn_nodes["general"]["hold_prob"]: 0.8,
                                        nn_nodes["supervised"]["s_LR"]: 0.0001})

                elif i < 30:
                    sess.run(nn_nodes["supervised"]["phase_network_train_fields"],
                             feed_dict={nn_nodes["supervised"]["x_in"]: batch_x,
                                        nn_nodes["supervised"]["actual_coefs_params"]: batch_y,
                                        nn_nodes["general"]["hold_prob"]: 0.8,
                                        nn_nodes["supervised"]["s_LR"]: 0.0001})

                else:
                    sess.run(nn_nodes["supervised"]["phase_network_train_phasecurve"],
                             feed_dict={nn_nodes["supervised"]["x_in"]: batch_x,
                                        nn_nodes["supervised"]["actual_coefs_params"]: batch_y,
                                        nn_nodes["general"]["hold_prob"]: 0.8,
                                        nn_nodes["supervised"]["s_LR"]: 0.0001})


            print("")

            add_tensorboard_values(nn_nodes, tf_loggers)

            # every x steps plot predictions
            if (i + 1) % 20 == 0 or (i + 1) <= 15:
                # update the plot

                update_plots(data_obj=get_data, sess=sess, nn_nodes=nn_nodes, modelname=modelname,
                             epoch=i+1, axes=axes,
                             streak_params=streak_params)

                # save model
                saver.save(sess, "models/" + modelname + ".ckpt")

            # return the index to 0
            get_data.batch_index = 0

        saver.save(sess, "models/" + modelname + ".ckpt")




