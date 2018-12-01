import crab_tf2
import shutil
import network2
import tensorflow as tf
import glob
import tables
import numpy as np
import  matplotlib.pyplot as plt
import os


def create_plots():

    fig = plt.figure(figsize=(7,10))
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

    # set phase angle to 0
    # find angle 0
    index_0_angle = 25
    #plt.figure(999)
    #plt.plot(np.real(actual_fields['xuv_f']), color='blue')
    #plt.plot(np.imag(actual_fields['xuv_f']), color='red')
    #plt.plot(np.unwrap(np.angle(actual_fields['xuv_f'])), color='green')
    #plt.plot([index_0_angle,index_0_angle], [-1, 1])
    #plt.ioff()
    #plt.show()

    actual_fields['xuv_f'] = actual_fields['xuv_f'] * np.exp(-1j * np.angle(actual_fields['xuv_f'][index_0_angle]))

    iteration = i+1

    # calculate loss
    loss_value = sess.run(network2.u_losses, feed_dict={network2.x: trace})
    print('iteration: {}'.format(iteration))
    print('loss_value: ', loss_value)

    axes['input_ir'].cla()
    axes['input_ir'].plot(np.real(actual_fields['ir_f']), color='blue')
    axes['input_ir'].plot(np.imag(actual_fields['ir_f']), color='red')
    axes['input_ir'].text(0.0, 1.05, 'actual IR', transform = axes['input_ir'].transAxes, backgroundcolor='white')
    axes['input_ir'].set_yticks([])
    axes['input_ir'].set_xticks([])

    axes['input_xuv'].cla()
    axes['input_xuv'].plot(np.real(actual_fields['xuv_f']), color='blue')
    axes['input_xuv'].plot(np.imag(actual_fields['xuv_f']), color='red')
    axes['input_xuv'].plot([index_0_angle, index_0_angle], [0.5,-0.5], alpha=0.3, linestyle='dashed', color='black')
    axes['input_xuv'].text(0.0, 1.05, 'actual XUV', transform=axes['input_xuv'].transAxes, backgroundcolor='white')
    axes['input_xuv'].set_yticks([])
    axes['input_xuv'].set_xticks([])

    axes['input_image'].cla()
    axes['input_image'].pcolormesh(input_image, cmap='jet')
    axes['input_image'].text(0.0, 1.05, 'input trace', transform=axes['input_image'].transAxes, backgroundcolor='white')
    axes['input_image'].set_yticks([])
    axes['input_image'].set_xticks([])


    axes['predicted_xuv'].cla()
    axes['predicted_xuv'].plot(np.real(predicted_fields['xuv_f']), color='blue')
    axes['predicted_xuv'].plot(np.imag(predicted_fields['xuv_f']), color='red')
    axes['predicted_xuv'].text(0.0, 1.05, 'predicted XUV', transform=axes['predicted_xuv'].transAxes, backgroundcolor='white')
    axes['predicted_xuv'].plot([index_0_angle, index_0_angle], [0.5, -0.5], alpha=0.3, linestyle='dashed', color='black')
    axes['predicted_xuv'].set_yticks([])
    axes['predicted_xuv'].set_xticks([])

    axes['predicted_ir'].cla()
    axes['predicted_ir'].plot(np.real(predicted_fields['ir_f']), color='blue')
    axes['predicted_ir'].plot(np.imag(predicted_fields['ir_f']), color='red')
    axes['predicted_ir'].text(0.0, 1.05, 'predicted IR', transform=axes['predicted_ir'].transAxes,
                               backgroundcolor='white')
    axes['predicted_ir'].set_yticks([])
    axes['predicted_ir'].set_xticks([])


    axes['generated_image'].cla()
    axes['generated_image'].pcolormesh(generated_image, cmap='jet')
    axes['generated_image'].text(0.0, 1.0, 'iteration: {}'.format(iteration), transform = axes['generated_image'].transAxes, backgroundcolor='white')
    axes['generated_image'].text(0.0, -0.1, 'streaking trace MSE: {}'.format(loss_value),
                                 transform=axes['generated_image'].transAxes, backgroundcolor='white')
    axes['generated_image'].set_yticks([])
    axes['generated_image'].set_xticks([])

    # save image
    dir = "./nnpictures/unsupervised/" + modelname + "/" + run_name + "/"
    if not os.path.isdir(dir):
        os.makedirs(dir)
    plt.savefig(dir + str(iteration) + ".png")


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
    writer.flush()


def generate_images_and_plot():
    # generate an image from the input image
    generated_image = sess.run(network2.image, feed_dict={network2.x: trace})
    trace_2d = trace.reshape(len(crab_tf2.p_values), len(crab_tf2.tau_values))
    predicted_fields_vector = sess.run(network2.y_pred, feed_dict={network2.x: trace})

    predicted_fields = {}
    predicted_fields['xuv_f'], predicted_fields['ir_f'] = network2.separate_xuv_ir_vec(predicted_fields_vector[0])

    # plot the trace and generated image
    update_plots(generated_image, trace_2d, actual_fields, predicted_fields)


# run name
# can do multiple run names for the same model
run_name = 'index1_5k_iterations_2'


# copy the model to a new version to use for unsupervised learning
modelname = 'unsupervised_11'
for file in glob.glob(r'./models/{}.ckpt.*'.format(modelname)):
    file_newname = file.replace(modelname, modelname+'_unsupervised')
    shutil.copy(file, file_newname)


# get the trace
trace, actual_fields = get_trace(index=1, filename='attstrace_test2_processed.hdf5')

# create mse tb measurer
unsupervised_mse_tb = tf.summary.scalar("streaking_trace_mse", network2.u_losses)


axes = create_plots()

plt.ion()
with tf.Session() as sess:

    writer = tf.summary.FileWriter("./tensorboard_graph_u/" + run_name)

    saver = tf.train.Saver()
    saver.restore(sess, './models/{}.ckpt'.format(modelname+'_unsupervised'))



    iterations = 5000
    for i in range(iterations):

        #learningrate = 0.00001 * (iterations/250)*0.1

        if (i + 1) % 20 == 0 or (i + 1) <= 15:

            generate_images_and_plot()


        # add tensorbaord values
        add_tensorboard_values()

        # train the network to reduce the error
        sess.run(network2.u_train, feed_dict={network2.x: trace, network2.u_LR: 0.00001})




