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


def separate_xuv_ir_vec(xuv_ir_vec):

    xuv = xuv_ir_vec[0:5]
    ir = xuv_ir_vec[5:]

    return xuv, ir


def plot_predictions(x_in, y_in, pred_in, indexes, axes, figure, epoch, set, net_name, nn_nodes, tf_generator_graphs, sess, streak_params):

    # get find where in the vector is the ir and xuv

    print("plot predicitons")


    for j, index in enumerate(indexes):

        prediction = pred_in[index]
        mse = sess.run(nn_nodes["loss"], feed_dict={nn_nodes["x"]: x_in[index].reshape(1, -1),nn_nodes["y_true"]: y_in[index].reshape(1, -1)})
        # print(mse)
        # print(str(mse))

        xuv_in, ir_in = separate_xuv_ir_vec(y_in[index])
        xuv_pred, ir_pred = separate_xuv_ir_vec(pred_in[index])

        xuv_in_Ef = sess.run(tf_generator_graphs["xuv_E_prop"]["f_cropped"], feed_dict={tf_generator_graphs["xuv_coefs_in"]: xuv_in.reshape(1, -1)})
        ir_in_Ef = sess.run(tf_generator_graphs["ir_E_prop"]["f_cropped"], feed_dict={tf_generator_graphs["ir_values_in"]: ir_in.reshape(1, -1)})

        xuv_pred_Ef = sess.run(tf_generator_graphs["xuv_E_prop"]["f_cropped"],feed_dict={tf_generator_graphs["xuv_coefs_in"]: xuv_pred.reshape(1, -1)})
        ir_pred_Ef = sess.run(tf_generator_graphs["ir_E_prop"]["f_cropped"],feed_dict={tf_generator_graphs["ir_values_in"]: ir_pred.reshape(1, -1)})

        xuv_in_Ef = xuv_in_Ef.reshape(-1)
        ir_in_Ef = ir_in_Ef.reshape(-1)
        xuv_pred_Ef = xuv_pred_Ef.reshape(-1)
        ir_pred_Ef = ir_pred_Ef.reshape(-1)


        axes[j]['input_trace'].cla()
        axes[j]['input_trace'].pcolormesh(x_in[index].reshape(len(streak_params["p_values"]), len(streak_params["tau_values"])), cmap='jet')
        axes[j]['input_trace'].text(0.0, 1.0, 'input_trace', transform=axes[j]['input_trace'].transAxes,backgroundcolor='white')
        axes[j]['input_trace'].set_xticks([])
        axes[j]['input_trace'].set_yticks([])

        axes[j]['actual_xuv'].cla()
        axes[j]['actual_xuv_twinx'].cla()
        axes[j]['actual_xuv'].plot(np.real(xuv_in_Ef), color='blue', alpha=0.3)
        axes[j]['actual_xuv'].plot(np.imag(xuv_in_Ef), color='red', alpha=0.3)
        axes[j]['actual_xuv'].plot(np.abs(xuv_in_Ef), color='black')
        # plot the phase
        axes[j]['actual_xuv_twinx'].plot(np.unwrap(np.angle(xuv_in_Ef)), color='green')
        axes[j]['actual_xuv_twinx'].tick_params(axis='y', colors='green')
        axes[j]['actual_xuv'].text(0.0,1.0, 'actual_xuv', transform=axes[j]['actual_xuv'].transAxes, backgroundcolor='white')
        axes[j]['actual_xuv'].set_xticks([])
        axes[j]['actual_xuv'].set_yticks([])

        axes[j]['predict_xuv'].cla()
        axes[j]['predict_xuv_twinx'].cla()
        axes[j]['predict_xuv'].plot(np.real(xuv_pred_Ef), color='blue', alpha=0.3)
        axes[j]['predict_xuv'].plot(np.imag(xuv_pred_Ef), color='red', alpha=0.3)
        axes[j]['predict_xuv'].plot(np.abs(xuv_pred_Ef), color='black')
        #plot the phase
        axes[j]['predict_xuv_twinx'].plot(np.unwrap(np.angle(xuv_pred_Ef)), color='green')
        axes[j]['predict_xuv_twinx'].tick_params(axis='y', colors='green')
        axes[j]['predict_xuv'].text(0.0, 1.0, 'predict_xuv', transform=axes[j]['predict_xuv'].transAxes, backgroundcolor='white')
        axes[j]['predict_xuv'].text(-0.4, 0, 'MSE: {} '.format(str(mse)),
                                   transform=axes[j]['predict_xuv'].transAxes, backgroundcolor='white')
        axes[j]['predict_xuv'].set_xticks([])
        axes[j]['predict_xuv'].set_yticks([])

        axes[j]['actual_ir'].cla()
        axes[j]['actual_ir'].plot(np.real(ir_in_Ef), color='blue')
        axes[j]['actual_ir'].plot(np.imag(ir_in_Ef), color='red')
        axes[j]['actual_ir'].text(0.0, 1.0, 'actual_ir', transform=axes[j]['actual_ir'].transAxes, backgroundcolor='white')

        if j == 0:

            axes[j]['actual_ir'].text(0.5, 1.25, net_name, transform=axes[j]['actual_ir'].transAxes,
                                      backgroundcolor='white')

            axes[j]['actual_ir'].text(0.5, 1.1, set, transform=axes[j]['actual_ir'].transAxes,
                                      backgroundcolor='white')

        axes[j]['actual_ir'].set_xticks([])
        axes[j]['actual_ir'].set_yticks([])

        axes[j]['predict_ir'].cla()
        axes[j]['predict_ir'].plot(np.real(ir_pred_Ef), color='blue')
        axes[j]['predict_ir'].plot(np.imag(ir_pred_Ef), color='red')
        axes[j]['predict_ir'].text(0.0, 1.0, 'predict_ir', transform=axes[j]['predict_ir'].transAxes,backgroundcolor='white')
        axes[j]['predict_ir'].set_xticks([])
        axes[j]['predict_ir'].set_yticks([])

        # calculate generated streaking trace
        generated_trace = sess.run(tf_generator_graphs["image"], feed_dict={tf_generator_graphs["ir_values_in"]: ir_pred.reshape(1, -1),
                                                              tf_generator_graphs["xuv_coefs_in"]: xuv_pred.reshape(1, -1)})

        axes[j]['reconstruct'].pcolormesh(generated_trace,cmap='jet')
        axes[j]['reconstruct'].text(0.0, 1.0, 'reconstructed_trace', transform=axes[j]['reconstruct'].transAxes,backgroundcolor='white')
        axes[j]['reconstruct'].set_xticks([])
        axes[j]['reconstruct'].set_yticks([])




        # save image
        dir = "./nnpictures/" + modelname + "/" + set + "/"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        figure.savefig(dir + str(epoch) + ".png")


def update_plots(data_obj, sess, nn_nodes, modelname, epoch, axes, tf_generator_graphs, streak_params):

    batch_x_train, batch_y_train = data_obj.evaluate_on_train_data(samples=500)
    predictions = sess.run(nn_nodes["y_pred"], feed_dict={nn_nodes["x"]: batch_x_train})


    plot_predictions(x_in=batch_x_train, y_in=batch_y_train, pred_in=predictions, indexes=[0, 1, 2],
                      axes=axes["trainplot1"], figure=axes["trainfig1"], epoch=epoch, set='train_data_1',
                      net_name=modelname, nn_nodes=nn_nodes, tf_generator_graphs=tf_generator_graphs, sess=sess,
                     streak_params=streak_params)

    plot_predictions(x_in=batch_x_train, y_in=batch_y_train, pred_in=predictions, indexes=[3, 4, 5],
                     axes=axes["trainplot2"], figure=axes["trainfig2"], epoch=epoch, set='train_data_2',
                     net_name=modelname, nn_nodes=nn_nodes, tf_generator_graphs=tf_generator_graphs, sess=sess,
                     streak_params=streak_params)

    batch_x_test, batch_y_test = data_obj.evaluate_on_test_data()
    predictions = sess.run(nn_nodes["y_pred"], feed_dict={nn_nodes["x"]: batch_x_test})

    plot_predictions(x_in=batch_x_test, y_in=batch_y_test, pred_in=predictions, indexes=[0, 1, 2],
                     axes=axes["testplot1"], figure=axes["testfig1"], epoch=epoch, set='test_data_1',
                     net_name=modelname, nn_nodes=nn_nodes, tf_generator_graphs=tf_generator_graphs, sess=sess,
                     streak_params=streak_params)

    plot_predictions(x_in=batch_x_test, y_in=batch_y_test, pred_in=predictions, indexes=[3, 4, 5],
                     axes=axes["testplot2"], figure=axes["testfig2"], epoch=epoch, set='test_data_2',
                     net_name=modelname, nn_nodes=nn_nodes, tf_generator_graphs=tf_generator_graphs, sess=sess,
                     streak_params=streak_params)

    plt.show()
    plt.pause(0.001)


def init_tf_loggers(nn_nodes):
    test_mse_tb = tf.summary.scalar("test_mse", nn_nodes["loss"])
    train_mse_tb = tf.summary.scalar("train_mse", nn_nodes["loss"])

    trackers = {}
    trackers["test_mse_tb"] = test_mse_tb
    trackers["train_mse_tb"] = train_mse_tb

    return trackers


def add_tensorboard_values(nn_nodes, tf_loggers):
    # view the mean squared error of the train data
    batch_x_test, batch_y_test = get_data.evaluate_on_test_data()
    print("test MSE: ", sess.run(nn_nodes["loss"], feed_dict={nn_nodes["x"]: batch_x_test, nn_nodes["y_true"]: batch_y_test}))
    summ = sess.run(tf_loggers["test_mse_tb"], feed_dict={nn_nodes["x"]: batch_x_test, nn_nodes["y_true"]: batch_y_test})
    writer.add_summary(summ, global_step=i + 1)

    # view the mean squared error of the train data
    batch_x_train, batch_y_train = get_data.evaluate_on_train_data(samples=500)
    print("train MSE: ", sess.run(nn_nodes["loss"], feed_dict={nn_nodes["x"]: batch_x_train, nn_nodes["y_true"]: batch_y_train}))
    summ = sess.run(tf_loggers["train_mse_tb"], feed_dict={nn_nodes["x"]: batch_x_train, nn_nodes["y_true"]: batch_y_train})
    writer.add_summary(summ, global_step=i + 1)

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

    # initialize IR generator
    # IR amplitudes
    amplitudes = {}
    amplitudes["phase_range"] = (0, 2 * np.pi)
    amplitudes["clambda_range"] = (1.6345, 1.6345)
    amplitudes["pulseduration_range"] = (7.0, 12.0)
    amplitudes["I_range"] = (0.4, 1.0)
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


def gan_network(input, output_length):

    with tf.variable_scope("gan"):
        hidden1 = tf.layers.dense(inputs=input, units=128)

        alpha = 0.01
        hidden1 = tf.maximum(alpha * hidden1, hidden1)

        hidden2 = tf.layers.dense(inputs=hidden1, units=128)

        hidden2 = tf.maximum(alpha * hidden2, hidden2)

        output = tf.layers.dense(hidden2, units=output_length, activation=tf.nn.tanh)

        return output


def phase_retrieval_net(input, total_label_length):

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
        # full_layer_one = normal_full_layer(convo_3_flat, 1024)
        full_layer_one = normal_full_layer(convo_3_flat, 2)
        print("layer needs to be set to 1024!!")

        # dropout
        hold_prob = tf.placeholder_with_default(1.0, shape=())
        dropout_layer = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

        y_pred = normal_full_layer(dropout_layer, total_label_length)

        return y_pred, hold_prob


def setup_neural_net(streak_params):

    xuv_phase_coefs = phase_parameters.params.xuv_phase_coefs

    print('Setting up multires layer network with more conv weights')


    # placeholders
    # x = tf.placeholder(tf.float32, shape=[None, int(len(streak_params["p_values"]) * len(streak_params["tau_values"]))])
    # define the label for supervised learning of phase retrieval net
    total_label_length = int(xuv_phase_coefs + 4)
    y_true = tf.placeholder(tf.float32, shape=[None, total_label_length])
    # define phase retrieval network

    # define GAN network
    gan_input = tf.placeholder(tf.float32, shape=[1, 100])
    # the output is one less than the xuv coefs, because linear phase will always be 0
    # add 4 because of the four IR parameters
    gan_output = gan_network(input=gan_input, output_length=xuv_phase_coefs-1 + 4)

    # GAN output is used to create XUV field and streaking trace
    # print("xuv_phase_coefs: ", xuv_phase_coefs)
    gan_xuv_out = gan_output[:, 0:xuv_phase_coefs-1]
    gan_ir_out = gan_output[:, xuv_phase_coefs-1:]
    # append a zero to the xuv gan out, corresponding to linear phase that is always 0
    samples_in = tf.shape(gan_xuv_out)[0]
    zeros_vec = tf.fill([samples_in, 1], 0.0)
    gan_xuv_out_nolin = tf.concat([zeros_vec, gan_xuv_out], axis=1)
    # use the gan outputs to generate fields

    # append to create label
    gan_label = tf.concat([gan_xuv_out_nolin, gan_ir_out], axis=1)
    xuv_E_prop = tf_functions.xuv_taylor_to_E(gan_xuv_out_nolin)
    # ir prop
    ir_E_prop = tf_functions.ir_from_params(gan_ir_out)
    # use the fields to generate streaking trace
    # sample size of one required as of now
    x, _ = tf_functions.streaking_trace(xuv_cropped_f_in=xuv_E_prop["f_cropped"][0],
                                            ir_cropped_f_in=ir_E_prop["f_cropped"][0])
    x_flat = tf.reshape(x, [1, -1])
    x_in = tf.placeholder_with_default(x_flat, shape=(None, int(len(streak_params["p_values"]) * len(streak_params["tau_values"]))))
    # pass image through network
    y_pred, hold_prob = phase_retrieval_net(input=x_in, total_label_length=total_label_length)


    # from y_pred generate fields
    xuv_params_pred = y_pred[:, 0:phase_parameters.params.xuv_phase_coefs]
    ir__params_pred = y_pred[:, phase_parameters.params.xuv_phase_coefs:]

    xuv_E_pred_prop = tf_functions.xuv_taylor_to_E(xuv_params_pred)
    ir_E_pred_prop = tf_functions.ir_from_params(ir__params_pred)

    reconstructed_trace = tf_functions.streaking_trace(xuv_cropped_f_in=xuv_E_pred_prop["f_cropped"][0],
                                            ir_cropped_f_in=ir_E_pred_prop["f_cropped"][0])


    # divide the variables to train with gan and phase retrieval net individually
    tvars = tf.trainable_variables()
    phase_net_vars = [var for var in tvars if "phase" in var.name]
    gan_net_vars = [var for var in tvars if "gan" in var.name]


    # define a loss function for GAN net reinforcement learning
    # print(gan_label)
    # print(gan_label[:, :xuv_phase_coefs])

    reward_scaler = tf.placeholder(tf.float32, shape=[])
    tf_ones_xuv = tf.ones_like(gan_label[:, :xuv_phase_coefs]) * reward_scaler
    tf_ones_ir = tf.ones_like(gan_label[:, xuv_phase_coefs:])
    reinf_label = tf.concat([tf_ones_xuv, tf_ones_ir], axis=1) * tf.stop_gradient(gan_label)
    reinf_loss = tf.losses.mean_squared_error(labels=reinf_label, predictions=gan_label)
    reinf_LR = tf.placeholder(tf.float32, shape=[])
    reinf_optimizer = tf.train.AdamOptimizer(learning_rate=reinf_LR)
    reinf_network_train = reinf_optimizer.minimize(reinf_loss, var_list=gan_net_vars)



    # loss function for training GAN network
    gan_network_loss = (1/tf.losses.mean_squared_error(labels=gan_label, predictions=y_pred))
    gan_LR = tf.placeholder(tf.float32, shape=[])
    gan_optimizer = tf.train.AdamOptimizer(learning_rate=gan_LR)
    gan_network_train = gan_optimizer.minimize(gan_network_loss, var_list=gan_net_vars)


    # loss function for training phase retrieval network
    phase_network_loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
    s_LR = tf.placeholder(tf.float32, shape=[])
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    phase_optimizer = tf.train.AdamOptimizer(learning_rate=s_LR)
    phase_network_train = phase_optimizer.minimize(phase_network_loss, var_list=phase_net_vars)



    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #
    #     images = np.zeros((5, 17458))
    #     for i in range(5):
    #         # gan_in = np.random.random(600).reshape(6, -1)
    #         gan_out = np.random.random(9).reshape(1, -1)
    #         # create images
    #         image = sess.run(x, feed_dict={gan_xuv_out_nolin: gan_out[:, 0:5],
    #                                        gan_ir_out: gan_out[:, 5:]})
    #         images[i, :] = image.reshape(-1)
    #
    #     out = sess.run(y_pred, feed_dict={x_in: images})
    #     print('1')
    #     print(np.shape(out))
    #
    #     gan_in = np.random.random(100).reshape(1, -1)
    #     out = sess.run(y_pred, feed_dict={gan_input: gan_in})
    #     print('2')
    #     print(np.shape(out))
    #
    #     out = sess.run(x, feed_dict={gan_input: gan_in})
    #     print('3')
    #     print(np.shape(out))


    # create graph for the unsupervised learning
    # xuv_cropped_f_tf, ir_cropped_f_tf = tf_seperate_xuv_ir_vec(y_pred)
    # image = crab_tf2.build_graph(xuv_cropped_f_in=xuv_cropped_f_tf, ir_cropped_f_in=ir_cropped_f_tf)
    # u_losses = tf.losses.mean_squared_error(labels=x, predictions=tf.reshape(image, [1, -1]))
    # u_LR = tf.placeholder(tf.float32, shape=[])
    # u_optimizer = tf.train.AdamOptimizer(learning_rate=u_LR)
    # u_train = u_optimizer.minimize(u_losses)

    nn_nodes = {}
    nn_nodes["gan"] = {}
    nn_nodes["supervised"] = {}
    nn_nodes["reconstruction"] = {}
    nn_nodes["reinforcement"] = {}

    # nodes specific to GAN training
    nn_nodes["gan"]["train"] = gan_network_train
    nn_nodes["gan"]["learningrate"] = gan_LR
    nn_nodes["gan"]["gan_label"] = gan_label
    nn_nodes["gan"]["gan_input"] = gan_input
    nn_nodes["gan"]["xuv_E_prop"] = xuv_E_prop
    nn_nodes["gan"]["ir_E_prop"] = ir_E_prop
    nn_nodes["gan"]["gan_xuv_out_nolin"] = gan_xuv_out_nolin

    # nodes specific to supervised learning
    nn_nodes["supervised"]["train"] = phase_network_train
    nn_nodes["supervised"]["learningrate"] = s_LR
    nn_nodes["supervised"]["trace_in"] = x_in
    nn_nodes["supervised"]["y_true"] = y_true
    nn_nodes["supervised"]["hold_prob"] = hold_prob

    # nodes specific to reinforcement learning
    nn_nodes["reinforcement"]["reward_scaler"] = reward_scaler
    nn_nodes["reinforcement"]["reinf_label"] = reinf_label
    nn_nodes["reinforcement"]["reinf_network_train"] = reinf_network_train
    nn_nodes["reinforcement"]["learningrate"] = reinf_LR

    # general nodes of network
    nn_nodes["trace_in"] = x_in
    nn_nodes["y_pred"] = y_pred

    # after y_pred, reconstruced pulse and fields
    nn_nodes["reconstruction"]["trace"] = reconstructed_trace
    nn_nodes["reconstruction"]["xuv_E_pred_prop"] = xuv_E_pred_prop
    nn_nodes["reconstruction"]["ir_E_pred_prop"] = ir_E_pred_prop

    return nn_nodes



if __name__ == "__main__":



    # initialize xuv, IR, and trace graphs
    tf_generator_graphs, streak_params = initialize_xuv_ir_trace_graphs()

    # build neural net graph
    nn_nodes = setup_neural_net(streak_params)

    # test gan output
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        fig = plt.figure()
        gs = fig.add_gridspec(2, 1)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        plt.ion()



        plotting = False
        indexmax = 2048 - 100
        indexmin = 100
        threshold_scaler = 1 / 250
        exceeded_threshold_plot = []
        avg_reward_plot = []


        for _ in range(1000):
            ax1.cla()
            ax2.cla()

            reward_values = []
            exceeded_threshold = []

            for _ in range(1000):

                gan_in = np.random.random(100).reshape(1, -1)

                out = sess.run(nn_nodes["gan"]["xuv_E_prop"]["t"],
                               feed_dict={nn_nodes["gan"]["gan_input"]: gan_in})
                threshold = threshold_scaler * np.max(np.abs(out[0]))
                indexmin_value = np.max(np.abs(out[0, :indexmin]))
                indexmax_value = np.max(np.abs(out[0, indexmax:]))
                reward = (2 * threshold) / (indexmin_value + indexmax_value)
                if reward > 1:
                    exceeded_threshold.append(reward)
                reward_values.append(reward)

            print("=====================")
            print("average reward:")
            print(len(reward_values))
            print(np.average(reward_values), "\n")
            avg_reward_plot.append(np.average(reward_values))


            print("reward < 1: ")
            print(len(exceeded_threshold), "\b")
            print("=====================")
            exceeded_threshold_plot.append(len(exceeded_threshold))

            ax1.plot(avg_reward_plot, color="blue")
            ax2.plot(exceeded_threshold_plot, color="red")
            plt.pause(0.001)

            for i in range(1000):

                gan_in = np.random.random(100).reshape(1, -1)

                out = sess.run(nn_nodes["gan"]["xuv_E_prop"]["t"],
                               feed_dict={nn_nodes["gan"]["gan_input"]: gan_in})

                threshold = threshold_scaler * np.max(np.abs(out[0]))
                indexmin_value = np.max(np.abs(out[0, :indexmin]))
                indexmax_value = np.max(np.abs(out[0, indexmax:]))
                reward = (2 * threshold) / (indexmin_value + indexmax_value)

                # train with the reward value
                sess.run(nn_nodes["reinforcement"]["reinf_network_train"],
                         feed_dict={nn_nodes["gan"]["gan_input"]: gan_in,
                                    nn_nodes["reinforcement"]["reward_scaler"]: reward,
                                    nn_nodes["reinforcement"]["learningrate"]: 0.00001})

        exit(0)



            # if plotting:
            #     ax.plot(np.real(out[0]), color="blue")
            #     ax.plot(np.abs(out[0]), color="black")
            #     # plot lines showing where the indexes are
            #     ax.plot([indexmin, indexmin], [indexmin_value, 0], color="red")
            #     ax.plot([indexmax, indexmax], [indexmax_value, 0], color="red")
            #     # plpt the threshold value
            #     ax.plot([indexmin, indexmax], [threshold, threshold], color="orange", linestyle="dashed")
            #     ax.plot([indexmin, indexmax], [0, 0], color="black", linestyle="dashed", alpha=0.5)
            #     ax.text(indexmin, 0.5*np.max(np.abs(out[0])), str(indexmin_value), backgroundcolor="white", ha='center')
            #     ax.text(indexmax, 0.5*np.max(np.abs(out[0])), str(indexmax_value), backgroundcolor="white", ha='center')
            #     ax.text(0.1, 0.1, "reward: {}".format(str(reward)), transform=ax.transAxes, backgroundcolor="white")
            #     plt.pause(1)




        # ganlabelinput = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        # ganlabelinput = np.array([[0.,0.01764787,0.28609192, -0.15447468, 0.25606465]])
        # out = sess.run(nn_nodes["gan"]["xuv_E_prop"]["t"],
        #                feed_dict={nn_nodes["gan"]["gan_xuv_out_nolin"]: orign_out[:,:5]})
        # ax = fig.add_subplot(gs[1, :])
        # ax.plot(np.real(out[0]), color="blue")
        # ax.plot(np.abs(out[0]), color="black")



    # test gan output
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        gan_in = np.random.random(100).reshape(1, -1)


        out = sess.run(nn_nodes["gan"]["gan_label"],
                       feed_dict={nn_nodes["gan"]["gan_input"]: gan_in})
        print("original output")
        print(out, "\n")


        out = sess.run(nn_nodes["reinforcement"]["reinf_label"],
                       feed_dict={nn_nodes["gan"]["gan_input"]: gan_in,
                                  nn_nodes["reinforcement"]["reward_scaler"]: 0.8})
        print("label")
        print(out, "\n")


        # train the network
        sess.run(nn_nodes["reinforcement"]["reinf_network_train"],
                       feed_dict={nn_nodes["gan"]["gan_input"]: gan_in,
                                  nn_nodes["reinforcement"]["reward_scaler"]: 0.8,
                                  nn_nodes["reinforcement"]["learningrate"]: 0.0001})


        out = sess.run(nn_nodes["gan"]["gan_label"],
                       feed_dict={nn_nodes["gan"]["gan_input"]: gan_in})

        print("new output")
        print(out, "\n")









        # out = sess.run(nn_nodes["gan"]["gan_label"],
        #                feed_dict={nn_nodes["gan"]["gan_input"]: gan_in})
        #
        # print(out)






        exit(0)




        # out = sess.run(nn_nodes["gan"]["gan_label"],
        #                feed_dict={nn_nodes["gan"]["gan_input"]: gan_in})
        #
        # print(out)
        #
        # out = sess.run(nn_nodes["gan"]["xuv_E_prop"]["t"],
        #          feed_dict={nn_nodes["gan"]["gan_input"]: gan_in})
        #
        # print(np.shape(out))
        # plt.figure(1)
        # plt.plot(np.real(out[0]), color="blue")
        # plt.plot(np.imag(out[0]), color="red")
        # plt.show()
        #
        # exit(0)


    print("built neural net")
    exit(0)


















    # init data object
    get_data = GetData(batch_size=10)

    # initialize mse tracking objects
    tf_loggers = init_tf_loggers(nn_nodes)


    # saver and set epoch number to run
    saver = tf.train.Saver()
    epochs = 900000

    # set the name of the neural net test run and save the settigns
    modelname = 'run3'

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
                if i < 35:
                    sess.run(nn_nodes["train"], feed_dict={nn_nodes["x"]: batch_x,
                                                           nn_nodes["y_true"]: batch_y,
                                                           nn_nodes["hold_prob"]: 0.8,
                                                           nn_nodes["s_LR"]: 0.0001})
                else:
                    sess.run(nn_nodes["train"], feed_dict={nn_nodes["x"]: batch_x,
                                                           nn_nodes["y_true"]: batch_y,
                                                           nn_nodes["hold_prob"]: 0.8,
                                                           nn_nodes["s_LR"]: 0.0001})
            print("")

            add_tensorboard_values(nn_nodes, tf_loggers)

            # every x steps plot predictions
            if (i + 1) % 20 == 0 or (i + 1) <= 15:
                # update the plot

                update_plots(data_obj=get_data, sess=sess, nn_nodes=nn_nodes, modelname=modelname,
                             epoch=i+1, axes=axes, tf_generator_graphs=tf_generator_graphs,
                             streak_params=streak_params)

                # save model
                saver.save(sess, "models/" + modelname + ".ckpt")

            # return the index to 0
            get_data.batch_index = 0

        saver.save(sess, "models/" + modelname + ".ckpt")




