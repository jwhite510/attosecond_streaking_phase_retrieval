import crab_tf2
import shutil
import network2
import tensorflow as tf
import glob
import tables
import numpy as np
import  matplotlib.pyplot as plt


def create_plots():

    fig = plt.figure()
    gs = fig.add_gridspec(4,2)

    axes = {}

    axes['input_image'] = fig.add_subplot(gs[1,:])
    axes['input_xuv'] = fig.add_subplot(gs[0,1])
    axes['input_ir'] = fig.add_subplot(gs[0,0])

    axes['generated_image'] = fig.add_subplot(gs[3,:])
    axes['predicted_xuv'] = fig.add_subplot(gs[2, 1])
    axes['predicted_ir'] = fig.add_subplot(gs[2, 0])

    return axes


def update_plots(generated_image, input_image, actual_fields, predicted_fields):

    iteration = i+1

    # calculate loss
    loss_value = sess.run(network2.u_losses, feed_dict={network2.x: trace})
    print('iteration: {}'.format(iteration))
    print('loss_value: ', loss_value)

    axes['input_ir'].cla()
    axes['input_ir'].plot(np.real(actual_fields['ir_f']), color='blue')
    axes['input_ir'].plot(np.imag(actual_fields['ir_f']), color='red')

    axes['input_xuv'].cla()
    axes['input_xuv'].plot(np.real(actual_fields['xuv_f']), color='blue')
    axes['input_xuv'].plot(np.imag(actual_fields['xuv_f']), color='red')

    axes['input_image'].cla()
    axes['input_image'].pcolormesh(input_image)


    axes['predicted_xuv'].cla()
    axes['predicted_xuv'].plot(np.real(predicted_fields['xuv_f']), color='blue')
    axes['predicted_xuv'].plot(np.imag(predicted_fields['xuv_f']), color='red')

    axes['predicted_ir'].cla()
    axes['predicted_ir'].plot(np.real(predicted_fields['ir_f']), color='blue')
    axes['predicted_ir'].plot(np.imag(predicted_fields['ir_f']), color='red')


    axes['generated_image'].cla()
    axes['generated_image'].pcolormesh(generated_image)
    axes['generated_image'].text(0.0, 1.0, 'iteration: {}'.format(iteration), transform = axes['generated_image'].transAxes, backgroundcolor='white')
    axes['generated_image'].text(0.0, 0.0, 'loss: {}'.format(loss_value),
                                 transform=axes['generated_image'].transAxes, backgroundcolor='white')

    plt.pause(0.001)


def get_trace(index, filename):

    with tables.open_file(filename, mode='r') as hdf5_file:

        trace = hdf5_file.root.trace[index,:]

        actual_fields = {}

        # actual_fields['ir_t'] = hdf5_file.root.ir_t[index,:]
        actual_fields['ir_f'] = hdf5_file.root.ir_f[index,:]

        # actual_fields['xuv_t'] = hdf5_file.root.xuv_t[index, :]
        actual_fields['xuv_f'] = hdf5_file.root.xuv_f[index, :]




    return trace.reshape(1,-1), actual_fields



def add_tensorboard_values():

    summ = sess.run(unsupervised_mse_tb, feed_dict={network2.x: trace})
    writer.add_summary(summ, global_step=i + 1)


# run name
# can do multiple run names for the same model
run_name = '1'


# copy the model to a new version to use for unsupervised learning
modelname = '2_test'
for file in glob.glob(r'./models/{}.ckpt.*'.format(modelname)):
    file_newname = file.replace(modelname, modelname+'_unsupervised')
    shutil.copy(file, file_newname)


# get the trace
trace, actual_fields = get_trace(index=0, filename='attstrace_test2_processed.hdf5')

# create mse tb measurer
unsupervised_mse_tb = tf.summary.scalar("test_mse", network2.u_losses)


axes = create_plots()

plt.ion()
with tf.Session() as sess:

    writer = tf.summary.FileWriter("./tensorboard_graph_u/" + run_name)

    saver = tf.train.Saver()
    saver.restore(sess, './models/{}.ckpt'.format(modelname+'_unsupervised'))



    iterations = 1000
    for i in range(iterations):

        # generate an image from the input image
        generated_image = sess.run(network2.image, feed_dict={network2.x: trace})
        trace_2d = trace.reshape(len(crab_tf2.p_values), len(crab_tf2.tau_values))
        predicted_fields_vector = sess.run(network2.y_pred, feed_dict={network2.x: trace})


        predicted_fields = {}
        predicted_fields['xuv_f'],predicted_fields['ir_f'] = network2.separate_xuv_ir_vec(predicted_fields_vector[0])

        # plot the trace and generated image
        update_plots(generated_image, trace_2d, actual_fields, predicted_fields)

        # add tensorbaord values
        add_tensorboard_values()

        # train the network to reduce the error
        sess.run(network2.u_train, feed_dict={network2.x: trace})




