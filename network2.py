import matplotlib.pyplot as plt
import tables
import numpy as np
import random
import tensorflow as tf
import crab_tf2
import os
import shutil


class GetData():
    def __init__(self, batch_size):

        self.batch_counter = 0
        self.batch_index = 0
        self.batch_size = batch_size
        self.train_filename = 'attstrace_train2_processed.hdf5'
        self.test_filename = 'attstrace_test2_processed.hdf5'

        # self.imagetype = 'proof'
        self.imagetype = 'rawtrace'

        self.labeltype = 'frequency'
        # self.labeltype = 'temporal'


        hdf5_file = tables.open_file(self.train_filename, mode="r")
        attstraces = hdf5_file.root.trace[:, :]
        self.samples = np.shape(attstraces)[0]

        hdf5_file.close()

    def next_batch(self):

        # retrieve the next batch of data from the data source
        hdf5_file = tables.open_file(self.train_filename, mode="r")


        if self.labeltype == 'frequency':
            xuv_real_batch = np.real(hdf5_file.root.xuv_f[self.batch_index:self.batch_index + self.batch_size, :])
            xuv_imag_batch = np.imag(hdf5_file.root.xuv_f[self.batch_index:self.batch_index + self.batch_size, :])
            xuv_appended_batch = np.append(xuv_real_batch, xuv_imag_batch, 1)

            ir_real_batch = np.real(hdf5_file.root.ir_f[self.batch_index:self.batch_index + self.batch_size, :])
            ir_imag_batch = np.imag(hdf5_file.root.ir_f[self.batch_index:self.batch_index + self.batch_size, :])
            ir_appended_batch = np.append(ir_real_batch, ir_imag_batch, 1)

            xuv_ir_appended = np.append(xuv_appended_batch, ir_appended_batch, 1)

            # xuv, ir = separate_xuv_ir_vec(xuv_ir_appended[0])

        if self.imagetype == 'rawtrace':
            trace_batch = hdf5_file.root.trace[self.batch_index:self.batch_index + self.batch_size, :]


        hdf5_file.close()

        self.batch_index += self.batch_size

        return  trace_batch, xuv_ir_appended


    def next_batch_random(self):

        pass

        # generate random indexes for the batch
        # make a vector of random integers between 0 and samples-1
        # indexes = random.sample(range(self.samples), self.batch_size)
        # hdf5_file = tables.open_file("processed.hdf5", mode="r")
        #
        # xuv_real_batch = np.real(hdf5_file.root.xuv_envelope[indexes, :])
        # xuv_imag_batch = np.imag(hdf5_file.root.xuv_envelope[indexes, :])
        # xuv_appended_batch = np.append(xuv_real_batch, xuv_imag_batch, 1)
        #
        # if self.imagetype == 'rawtrace':
        #     trace_batch = hdf5_file.root.attstrace[indexes, :]
        # elif self.imagetype == 'proof':
        #     trace_batch = hdf5_file.root.proof[indexes, :]
        #
        #
        # hdf5_file.close()
        #
        # self.batch_index += self.batch_size
        #
        # return  trace_batch, xuv_appended_batch


    def evaluate_on_test_data(self):

        # this is used to evaluate the mean squared error of the data after every epoch
        hdf5_file = tables.open_file(self.test_filename, mode="r")

        if self.labeltype == 'frequency':

            xuv_real_eval = np.real(hdf5_file.root.xuv_f[:, :])
            xuv_imag_eval = np.imag(hdf5_file.root.xuv_f[:, :])
            xuv_appended_eval = np.append(xuv_real_eval, xuv_imag_eval, 1)

            ir_real_eval = np.real(hdf5_file.root.ir_f[:, :])
            ir_imag_eval = np.imag(hdf5_file.root.ir_f[:, :])
            ir_appended_eval = np.append(ir_real_eval, ir_imag_eval, 1)

            xuv_ir_appended = np.append(xuv_appended_eval, ir_appended_eval, 1)



        if self.imagetype == 'rawtrace':
            trace_eval = hdf5_file.root.trace[:, :]

        hdf5_file.close()

        return trace_eval, xuv_ir_appended



    def evaluate_on_train_data(self, samples):

        # this is used to evaluate the mean squared error of the data after every epoch
        hdf5_file = tables.open_file(self.train_filename, mode="r")

        if self.labeltype == 'frequency':
            xuv_real_eval = np.real(hdf5_file.root.xuv_f[:samples, :])
            xuv_imag_eval = np.imag(hdf5_file.root.xuv_f[:samples, :])
            xuv_appended_eval = np.append(xuv_real_eval, xuv_imag_eval, 1)

            ir_real_eval = np.real(hdf5_file.root.ir_f[:samples, :])
            ir_imag_eval = np.imag(hdf5_file.root.ir_f[:samples, :])
            ir_appended_eval = np.append(ir_real_eval, ir_imag_eval, 1)

            xuv_ir_appended = np.append(xuv_appended_eval, ir_appended_eval, 1)



        if self.imagetype == 'rawtrace':
            trace_eval = hdf5_file.root.trace[:samples, :]

        hdf5_file.close()

        return trace_eval, xuv_ir_appended


def plot_predictions(x_in, y_in, axis, fig, set, modelname, epoch, inputtype):

    mses = []
    predictions = sess.run(y_pred, feed_dict={x: x_in,
                                              y_true: y_in})

    for ax, index in zip([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]):

        mse = sess.run(loss, feed_dict={x: x_in[index].reshape(1, -1),
                                        y_true: y_in[index].reshape(1, -1)})
        mses.append(mse)

        # plot  actual trace
        axis[0][ax].pcolormesh(x_in[index].reshape(len(crab_tf2.p_values),
                                                   len(crab_tf2.tau_values)), cmap='jet')

        if inputtype == 'rawtrace':
            axis[0][ax].text(0.1, 1, "unprocessed trace", transform=axis[0][ax].transAxes,
                             backgroundcolor='white')
        elif inputtype == 'proof':
            axis[0][ax].text(0.1, 1, "PROOF", transform=axis[0][ax].transAxes,
                             backgroundcolor='white')

        axis[0][ax].text(0.1, 1.1, "Epoch : {}".format(i + 1),
                         transform=axis[0][ax].transAxes, backgroundcolor='white')

        # plot E(t) retrieved
        axis[1][ax].cla()
        axis[1][ax].plot(predictions[index, :64], color="blue")
        axis[1][ax].plot(predictions[index, 64:], color="red")

        axis[1][ax].text(0.1, 1.1, "MSE: " + str(mse),
                         transform=axis[1][ax].transAxes, backgroundcolor='white')
        axis[1][ax].text(0.1, 1, "prediction [" + set + " set]", transform=axis[1][ax].transAxes,
                         backgroundcolor='white')

        # plot E(t) actual
        axis[2][ax].cla()
        axis[2][ax].plot(y_in[index, :64], color="blue")
        axis[2][ax].plot(y_in[index, 64:], color="red")
        axis[2][ax].text(0.1, 1, "actual [" + set + " set]", transform=axis[2][ax].transAxes,
                         backgroundcolor='white')


        axis[0][ax].set_xticks([])
        axis[0][ax].set_yticks([])
        axis[1][ax].set_xticks([])
        axis[1][ax].set_yticks([])
        axis[2][ax].set_xticks([])
        axis[2][ax].set_yticks([])


    print("mses: ", mses)
    print("avg : ", (1 / len(mses)) * np.sum(np.array(mses)))

    # save image
    dir = "./nnpictures/" + modelname + "/" + set + "/"
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig.savefig(dir + str(epoch) + ".png")


def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(init_bias_vals)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding='SAME')


def convolutional_layer(input_x, shape, activate, stride):
    W = init_weights(shape)
    b = init_bias([shape[3]])

    if activate == 'relu':
        return tf.nn.relu(conv2d(input_x, W, stride) + b)

    if activate == 'leaky':
        return tf.nn.leaky_relu(conv2d(input_x, W, stride) + b)

    elif activate == 'none':
        return conv2d(input_x, W, stride) + b


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


def multires_layer(input, input_channels, filter_sizes):

    # list of layers
    filters = []
    for filter_size in filter_sizes:
        # create filter
        filters.append(convolutional_layer(input, shape=[filter_size, filter_size,
                        input_channels, input_channels], activate='relu', stride=[1, 1]))

    concat_layer = tf.concat(filters, axis=3)
    return concat_layer


def create_sample_plot(samples_per_plot=3):

    fig = plt.figure(figsize=(14, 8))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05,
                            wspace=0.1, hspace=0.1)
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


def plot_predictions2(x_in, y_in, pred_in, indexes, axes, figure, epoch, set):

    # get find where in the vector is the ir and xuv


    for j, index in enumerate(indexes):

        prediction = pred_in[index]
        mse = sess.run(loss, feed_dict={x: x_in[index].reshape(1, -1),y_true: y_in[index].reshape(1, -1)})
        # print(mse)
        # print(str(mse))


        xuv_in, ir_in = separate_xuv_ir_vec(y_in[index])
        xuv_pred, ir_pred = separate_xuv_ir_vec(pred_in[index])


        axes[j]['input_trace'].cla()
        axes[j]['input_trace'].pcolormesh(x_in[index].reshape(len(crab_tf2.p_values), len(crab_tf2.tau_values)), cmap='jet')
        axes[j]['input_trace'].text(0.0, 1.0, 'input_trace', transform=axes[j]['input_trace'].transAxes,backgroundcolor='white')
        axes[j]['input_trace'].set_xticks([])
        axes[j]['input_trace'].set_yticks([])

        axes[j]['actual_xuv'].cla()
        axes[j]['actual_xuv_twinx'].cla()
        axes[j]['actual_xuv'].plot(np.real(xuv_in), color='blue', alpha=0.3)
        axes[j]['actual_xuv'].plot(np.imag(xuv_in), color='red', alpha=0.3)
        axes[j]['actual_xuv'].plot(np.abs(xuv_in), color='black')
        # plot the phase
        axes[j]['actual_xuv_twinx'].plot(np.unwrap(np.angle(xuv_in)), color='green')
        axes[j]['actual_xuv_twinx'].tick_params(axis='y', colors='green')
        axes[j]['actual_xuv'].text(0.0,1.0, 'actual_xuv', transform=axes[j]['actual_xuv'].transAxes, backgroundcolor='white')
        axes[j]['actual_xuv'].set_xticks([])
        axes[j]['actual_xuv'].set_yticks([])

        axes[j]['predict_xuv'].cla()
        axes[j]['predict_xuv_twinx'].cla()
        axes[j]['predict_xuv'].plot(np.real(xuv_pred), color='blue', alpha=0.3)
        axes[j]['predict_xuv'].plot(np.imag(xuv_pred), color='red', alpha=0.3)
        axes[j]['predict_xuv'].plot(np.abs(xuv_pred), color='black')
        #plot the phase
        axes[j]['predict_xuv_twinx'].plot(np.unwrap(np.angle(xuv_pred)), color='green')
        axes[j]['predict_xuv_twinx'].tick_params(axis='y', colors='green')
        axes[j]['predict_xuv'].text(0.0, 1.0, 'predict_xuv', transform=axes[j]['predict_xuv'].transAxes, backgroundcolor='white')
        axes[j]['predict_xuv'].text(-0.4, 0, 'MSE: {} '.format(str(mse)),
                                   transform=axes[j]['predict_xuv'].transAxes, backgroundcolor='white')
        axes[j]['predict_xuv'].set_xticks([])
        axes[j]['predict_xuv'].set_yticks([])

        axes[j]['actual_ir'].cla()
        axes[j]['actual_ir'].plot(np.real(ir_in), color='blue')
        axes[j]['actual_ir'].plot(np.imag(ir_in), color='red')
        axes[j]['actual_ir'].text(0.0, 1.0, 'actual_ir', transform=axes[j]['actual_ir'].transAxes, backgroundcolor='white')
        axes[j]['actual_ir'].set_xticks([])
        axes[j]['actual_ir'].set_yticks([])

        axes[j]['predict_ir'].cla()
        axes[j]['predict_ir'].plot(np.real(ir_pred), color='blue')
        axes[j]['predict_ir'].plot(np.imag(ir_pred), color='red')
        axes[j]['predict_ir'].text(0.0, 1.0, 'predict_ir', transform=axes[j]['predict_ir'].transAxes,backgroundcolor='white')
        axes[j]['predict_ir'].set_xticks([])
        axes[j]['predict_ir'].set_yticks([])

        # calculate generated streaking trace
        generated_trace = sess.run(crab_tf2.image, feed_dict={crab_tf2.ir_cropped_f: ir_pred,
                                                              crab_tf2.xuv_cropped_f: xuv_pred})

        axes[j]['reconstruct'].pcolormesh(generated_trace,cmap='jet')
        axes[j]['reconstruct'].text(0.0, 1.0, 'reconstructed_trace', transform=axes[j]['reconstruct'].transAxes,backgroundcolor='white')
        axes[j]['reconstruct'].set_xticks([])
        axes[j]['reconstruct'].set_yticks([])




        # save image
        dir = "./nnpictures/" + modelname + "/" + set + "/"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        figure.savefig(dir + str(epoch) + ".png")


def update_plots():

    batch_x_train, batch_y_train = get_data.evaluate_on_train_data(samples=500)
    predictions = sess.run(y_pred, feed_dict={x: batch_x_train})

    plot_predictions2(x_in=batch_x_train, y_in=batch_y_train, pred_in=predictions, indexes=[0, 1, 2],
                      axes=trainplot1, figure=trainfig1, epoch=i + 1, set='train_data_1')

    plot_predictions2(x_in=batch_x_train, y_in=batch_y_train, pred_in=predictions, indexes=[3, 4, 5],
                      axes=trainplot2, figure=trainfig2, epoch=i + 1, set='train_data_2')

    batch_x_test, batch_y_test = get_data.evaluate_on_test_data()
    predictions = sess.run(y_pred, feed_dict={x: batch_x_test})

    plot_predictions2(x_in=batch_x_test, y_in=batch_y_test, pred_in=predictions, indexes=[0, 1, 2],
                      axes=testplot1, figure=testfig1, epoch=i + 1, set='test_data_1')

    plot_predictions2(x_in=batch_x_test, y_in=batch_y_test, pred_in=predictions, indexes=[3, 4, 5],
                      axes=testplot2, figure=testfig2, epoch=i + 1, set='test_data_2')

    plt.show()
    plt.pause(0.001)


def add_tensorboard_values():
    # view the mean squared error of the train data
    batch_x_test, batch_y_test = get_data.evaluate_on_test_data()
    print("test MSE: ", sess.run(loss, feed_dict={x: batch_x_test, y_true: batch_y_test}))
    summ = sess.run(test_mse_tb, feed_dict={x: batch_x_test, y_true: batch_y_test})
    writer.add_summary(summ, global_step=i + 1)

    # view the mean squared error of the train data
    batch_x_train, batch_y_train = get_data.evaluate_on_train_data(samples=500)
    print("train MSE: ", sess.run(loss, feed_dict={x: batch_x_train, y_true: batch_y_train}))
    summ = sess.run(train_mse_tb, feed_dict={x: batch_x_train, y_true: batch_y_train})
    writer.add_summary(summ, global_step=i + 1)

    writer.flush()


def show_loading_bar():
    global dots
    # display loading bar
    percent = 50 * get_data.batch_index / get_data.samples
    if percent - dots > 1:
        print(".", end="", flush=True)
        dots += 1


def separate_xuv_ir_vec(xuv_ir_vec):


    xuv_real = xuv_ir_vec[:int(xuv_input_length/2)]
    xuv_imag = xuv_ir_vec[int(xuv_input_length/2):int(xuv_input_length)]

    ir_real = xuv_ir_vec[int(xuv_input_length):int(xuv_input_length+(ir_input_length/2))]
    ir_imag = xuv_ir_vec[int(xuv_input_length+(ir_input_length/2)):]

    xuv = xuv_real + 1j * xuv_imag
    ir = ir_real + 1j * ir_imag


    return xuv, ir


def tf_seperate_xuv_ir_vec(tensor):

    xuv_real = tensor[0][:int(xuv_input_length / 2)]
    xuv_imag = tensor[0][int(xuv_input_length / 2):int(xuv_input_length)]

    ir_real = tensor[0][int(xuv_input_length):int(xuv_input_length + (ir_input_length / 2))]
    ir_imag = tensor[0][int(xuv_input_length + (ir_input_length / 2)):]

    xuv = tf.complex(real=xuv_real, imag=xuv_imag)
    ir = tf.complex(real=ir_real, imag=ir_imag)

    return xuv, ir




# placeholders
x = tf.placeholder(tf.float32, shape=[None, int(len(crab_tf2.p_values)*len(crab_tf2.tau_values))])

xuv_input_length = int(len(crab_tf2.xuv.Ef_prop_cropped)*2)
ir_input_length = int(len(crab_tf2.ir.Ef_prop_cropped)*2)
total_input_length = xuv_input_length + ir_input_length
y_true = tf.placeholder(tf.float32, shape=[None, total_input_length])


#input image
x_image = tf.reshape(x, [-1, len(crab_tf2.p_values), len(crab_tf2.tau_values), 1])

network = 1

"""
network 1 uses a 3 convolutional layers followed by two dense layers

network 2 uses the multires layer setting
"""
if network == 1:

    print('Setting up standard convolutional network')

    # shape = [sizex, sizey, channels, filters/features]
    convo_1 = convolutional_layer(x_image, shape=[4, 4, 1, 32], activate='none', stride=[2, 2])
    convo_2 = convolutional_layer(convo_1, shape=[2, 2, 32, 32], activate='none', stride=[2, 2])
    convo_3 = convolutional_layer(convo_2, shape=[1, 1, 32, 32], activate='leaky', stride=[1, 1])

    #convo_3_flat = tf.reshape(convo_3, [-1, 58*106*32])
    convo_3_flat = tf.contrib.layers.flatten(convo_3)
    #full_layer_one = tf.nn.relu(normal_full_layer(convo_3_flat, 512))
    full_layer_one = normal_full_layer(convo_3_flat, 512)

    #dropout
    hold_prob = tf.placeholder_with_default(tf.constant(1.0, dtype=tf.float32), shape=())
    dropout_layer = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)


    y_pred = normal_full_layer(dropout_layer, total_input_length)
    #y_pred = normal_full_layer(full_layer_one, 128)

    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss)

    # create graph for the unsupervised learning
    xuv_cropped_f_tf, ir_cropped_f_tf = tf_seperate_xuv_ir_vec(y_pred)
    image = crab_tf2.build_graph(xuv_cropped_f_in=xuv_cropped_f_tf, ir_cropped_f_in=ir_cropped_f_tf)
    u_losses = tf.losses.mean_squared_error(labels=x, predictions=tf.reshape(image, [1, -1]))
    u_LR = tf.placeholder( tf.float32, shape=[])
    u_optimizer = tf.train.AdamOptimizer(learning_rate=u_LR)
    u_train = u_optimizer.minimize(u_losses)


elif network == 2:

    print('Setting up multires layer network')

    # six convolutional layers
    convolutional_outputs = 16

    multires_filters = [11, 7, 5, 3]

    multires_layer_1 = multires_layer(input=x_image, input_channels=1, filter_sizes=multires_filters)

    conv_layer_1= convolutional_layer(multires_layer_1, shape=[1, 1, len(multires_filters), convolutional_outputs],
                            activate='relu', stride=[2, 2])



    multires_layer_2 = multires_layer(input=conv_layer_1, input_channels=convolutional_outputs,
                                      filter_sizes=multires_filters)

    conv_layer_2 = convolutional_layer(multires_layer_2, shape=[1, 1, int(convolutional_outputs*len(multires_filters)),
                                convolutional_outputs], activate='relu', stride=[2, 2])



    multires_layer_3 = multires_layer(input=conv_layer_2, input_channels=convolutional_outputs,
                                      filter_sizes=multires_filters)

    conv_layer_3 = convolutional_layer(multires_layer_3,
                                       shape=[1, 1, int(convolutional_outputs * len(multires_filters)),
                                              convolutional_outputs], activate='relu', stride=[2, 2])

    convo_3_flat = tf.contrib.layers.flatten(conv_layer_3)
    full_layer_one = normal_full_layer(convo_3_flat, 512)

    # dropout
    hold_prob = tf.placeholder_with_default(1.0, shape=())
    dropout_layer = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

    y_pred = normal_full_layer(dropout_layer, 128)

    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss)


if __name__ == "__main__":

    init = tf.global_variables_initializer()

    # initialize data object
    get_data = GetData(batch_size=10)


    # initialize mse tracking objects
    test_mse_tb = tf.summary.scalar("test_mse", loss)
    train_mse_tb = tf.summary.scalar("train_mse", loss)


    # saver and set epoch number to run
    saver = tf.train.Saver()
    epochs = 200

    # set the name of the neural net test run and save the settigns
    modelname = 'gdd_larger_tmax_default_ir_gddtod_phaseplots'
    print('starting ' + modelname)
    # save this file
    shutil.copyfile('./network2.py', './models/network2_{}.py'.format(modelname))


    # create figures for showing results
    testplot1, testfig1 = create_sample_plot()
    testplot2, testfig2 = create_sample_plot()

    trainplot1, trainfig1 = create_sample_plot()
    trainplot2, trainfig2 = create_sample_plot()

    plt.ion()

    with tf.Session() as sess:

        sess.run(init)

        writer = tf.summary.FileWriter("./tensorboard_graph/" + modelname)

        for i in range(epochs):

            print("Epoch : {}".format(i+1))

            # iterate through every sample in the training set
            dots = 0
            while get_data.batch_index < get_data.samples:

                show_loading_bar()

                # retrieve data
                batch_x, batch_y = get_data.next_batch()

                # retrieve random samples
                #batch_x, batch_y = get_data.next_batch_random()

                #train network
                sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.8})

            print("")

            add_tensorboard_values()

            # every x steps plot predictions
            if (i + 1) % 20 == 0 or (i + 1) <= 15:

                # update the plot
                update_plots()

            # return the index to 0
            get_data.batch_index = 0


        saver.save(sess, "models/" + modelname + ".ckpt")













