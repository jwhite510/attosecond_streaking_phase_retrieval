import matplotlib.pyplot as plt
import tables
import numpy as np
import random
import tensorflow as tf
import generate_proof_traces
import os
import shutil

class GetData():
    def __init__(self, batch_size):

        self.batch_counter = 0
        self.batch_index = 0
        self.batch_size = batch_size
        self.train_filename = 'attstrace_train_processed.hdf5'
        self.test_filename = 'attstrace_test_processed.hdf5'

        # self.imagetype = 'proof'
        self.imagetype = 'rawtrace'

        self.labeltype = 'frequency'
        # self.labeltype = 'temporal'


        hdf5_file = tables.open_file(self.train_filename, mode="r")
        attstraces = hdf5_file.root.attstrace[:, :]
        self.samples = np.shape(attstraces)[0]
        hdf5_file.close()

    def next_batch(self):

        # retrieve the next batch of data from the data source
        hdf5_file = tables.open_file(self.train_filename, mode="r")


        if self.labeltype == 'temporal':
            xuv_real_batch = np.real(hdf5_file.root.xuv_envelope[self.batch_index:self.batch_index+self.batch_size, :])
            xuv_imag_batch = np.imag(hdf5_file.root.xuv_envelope[self.batch_index:self.batch_index+self.batch_size, :])
            xuv_appended_batch = np.append(xuv_real_batch, xuv_imag_batch, 1)

        elif self.labeltype == 'frequency':
            xuv_real_batch = np.real(hdf5_file.root.xuv_frequency_domain[self.batch_index:self.batch_index + self.batch_size, :])
            xuv_imag_batch = np.imag(hdf5_file.root.xuv_frequency_domain[self.batch_index:self.batch_index + self.batch_size, :])
            xuv_appended_batch = np.append(xuv_real_batch, xuv_imag_batch, 1)



        if self.imagetype == 'rawtrace':
            trace_batch = hdf5_file.root.attstrace[self.batch_index:self.batch_index + self.batch_size, :]

        elif self.imagetype == 'proof':
            trace_batch = hdf5_file.root.proof[self.batch_index:self.batch_index+self.batch_size, :]


        hdf5_file.close()

        self.batch_index += self.batch_size

        return  trace_batch, xuv_appended_batch


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

        if self.labeltype == 'temporal':

            xuv_real_eval = np.real(hdf5_file.root.xuv_envelope[:, :])
            xuv_imag_eval = np.imag(hdf5_file.root.xuv_envelope[:, :])
            xuv_appended_eval = np.append(xuv_real_eval, xuv_imag_eval, 1)

        elif self.labeltype == 'frequency':

            xuv_real_eval = np.real(hdf5_file.root.xuv_frequency_domain[:, :])
            xuv_imag_eval = np.imag(hdf5_file.root.xuv_frequency_domain[:, :])
            xuv_appended_eval = np.append(xuv_real_eval, xuv_imag_eval, 1)


        if self.imagetype == 'rawtrace':
            trace_eval = hdf5_file.root.attstrace[:, :]
        elif self.imagetype == 'proof':
            trace_eval = hdf5_file.root.proof[:, :]

        hdf5_file.close()

        return trace_eval, xuv_appended_eval



    def evaluate_on_train_data(self, samples):

        # this is used to evaluate the mean squared error of the data after every epoch
        hdf5_file = tables.open_file(self.train_filename, mode="r")

        if self.labeltype == 'temporal':
            xuv_real_eval = np.real(hdf5_file.root.xuv_envelope[:samples, :])
            xuv_imag_eval = np.imag(hdf5_file.root.xuv_envelope[:samples, :])
            xuv_appended_eval = np.append(xuv_real_eval, xuv_imag_eval, 1)

        elif self.labeltype == 'frequency':
            xuv_real_eval = np.real(hdf5_file.root.xuv_frequency_domain[:samples, :])
            xuv_imag_eval = np.imag(hdf5_file.root.xuv_frequency_domain[:samples, :])
            xuv_appended_eval = np.append(xuv_real_eval, xuv_imag_eval, 1)

        if self.imagetype == 'rawtrace':
            trace_eval = hdf5_file.root.attstrace[:samples, :]
        elif self.imagetype == 'proof':
            trace_eval = hdf5_file.root.proof[:samples, :]

        hdf5_file.close()

        return trace_eval, xuv_appended_eval



def plot_predictions(x_in, y_in, axis, fig, set, modelname, epoch, inputtype):

    mses = []
    predictions = sess.run(y_pred, feed_dict={x: x_in,
                                              y_true: y_in})

    for ax, index in zip([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]):

        mse = sess.run(loss, feed_dict={x: x_in[index].reshape(1, -1),
                                        y_true: y_in[index].reshape(1, -1)})
        mses.append(mse)

        # plot  actual trace
        axis[0][ax].pcolormesh(x_in[index].reshape(len(generate_proof_traces.p_vec),
                                                   len(generate_proof_traces.tauvec)), cmap='jet')

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
    dir = "/home/zom/PythonProjects/attosecond_streaking_phase_retrieval/nnpictures/" + modelname + "/" + set + "/"
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig.savefig(dir + str(epoch) + ".png")





def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding='SAME')


# convolutional layer
def convolutional_layer(input_x, shape, activate, stride):
    W = init_weights(shape)
    b = init_bias([shape[3]])

    if activate == 'relu':
        return tf.nn.relu(conv2d(input_x, W, stride) + b)

    if activate == 'leaky':
        return tf.nn.leaky_relu(conv2d(input_x, W, stride) + b)

    elif activate == 'none':
        return conv2d(input_x, W, stride) + b


# dense layer
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





# placeholders
x = tf.placeholder(tf.float32, shape=[None, int(len(generate_proof_traces.p_vec)*len(generate_proof_traces.tauvec))])
y_true = tf.placeholder(tf.float32, shape=[None, int(generate_proof_traces.xuv_field_length*2)])


#input image
x_image = tf.reshape(x, [-1, len(generate_proof_traces.p_vec), len(generate_proof_traces.tauvec), 1])

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
    hold_prob = tf.placeholder_with_default(1.0, shape=())
    dropout_layer = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)


    y_pred = normal_full_layer(dropout_layer, 128)
    #y_pred = normal_full_layer(full_layer_one, 128)

    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss)


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

    test_mse_tb = tf.summary.scalar("test_mse", loss)

    train_mse_tb = tf.summary.scalar("train_mse", loss)

    saver = tf.train.Saver()
    epochs = 200

    modelname = 'reg_conv_net_1'
    print('starting ' + modelname)
    # save this file
    shutil.copyfile('./network.py', './models/network_{}.py'.format(modelname))

    fig1, ax1 = plt.subplots(3, 6, figsize=(14, 8))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05,
                        wspace=0.1, hspace=0.1)

    fig2, ax2 = plt.subplots(3, 6, figsize=(14, 8))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05,
                        wspace=0.1, hspace=0.1)

    plt.ion()

    with tf.Session() as sess:
        sess.run(init)

        writer = tf.summary.FileWriter("./tensorboard_graph/" + modelname)


        # print('hello')
        # images = sess.run(x_image, feed_dict={x:batch_x})

        for i in range(epochs):
            print("Epoch : {}".format(i+1))

            # iterate through every sample in the training set
            dots = 0
            while get_data.batch_index < get_data.samples:

                # display loading bar
                percent = 50 * get_data.batch_index / get_data.samples
                if percent - dots > 1:
                    print(".", end="", flush=True)
                    dots += 1

                # retrieve data
                batch_x, batch_y = get_data.next_batch()

                # retrieve random samples
                #batch_x, batch_y = get_data.next_batch_random()

                #train network
                sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.8})

            print("")

            # view the mean squared error of the train data
            batch_x_test, batch_y_test = get_data.evaluate_on_test_data()
            print("test MSE: ", sess.run(loss, feed_dict={x: batch_x_test, y_true: batch_y_test}))
            summ = sess.run(test_mse_tb, feed_dict={x: batch_x_test, y_true: batch_y_test})
            writer.add_summary(summ, global_step=i + 1)

            # view the mean squared error of the train data
            batch_x_train, batch_y_train = get_data.evaluate_on_train_data(samples=500)
            print("train MSE: ", sess.run(loss, feed_dict={x: batch_x_train, y_true: batch_y_train}))
            summ = sess.run(train_mse_tb, feed_dict={x: batch_x_train, y_true: batch_y_train})
            writer.add_summary(summ, global_step=i+1)

            writer.flush()

            # every x steps plot predictions
            if (i + 1) % 20 == 0 or (i + 1) <= 15:
                # update the plot

                batch_x_train, batch_y_train = get_data.evaluate_on_train_data(samples=500)
                plot_predictions(x_in=batch_x_train, y_in=batch_y_train, axis=ax1, fig=fig1,
                                 set="train", modelname=modelname, epoch=i + 1, inputtype=get_data.imagetype)


                batch_x_test, batch_y_test = get_data.evaluate_on_test_data()
                plot_predictions(x_in=batch_x_test, y_in=batch_y_test, axis=ax2, fig=fig2,
                                 set="test", modelname=modelname, epoch=i + 1, inputtype=get_data.imagetype)


                plt.show()
                plt.pause(0.001)


            # return the index to 0
            get_data.batch_index = 0



        saver.save(sess, "models/" + modelname + ".ckpt")













