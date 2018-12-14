import crab_tf2
xuv_time_domain_func = crab_tf2.xuv_time_domain
import shutil
import network2
import tensorflow as tf
import glob
import tables
import numpy as np
import  matplotlib.pyplot as plt
import os
import csv
from scipy import interpolate
import scipy.constants as sc



def get_measured_trace():
    filepath = './experimental_data/53asstreakingdata.csv'
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        matrix = np.array(list(reader))

        Energy = matrix[4:, 0].astype('float')
        Delay = matrix[2, 2:].astype('float')
        Values = matrix[4:, 2:].astype('float')


    # map the function onto a grid matching the training data
    interp2 = interpolate.interp2d(Delay, Energy, Values, kind='linear')
    timespan = np.abs(Delay[-1]) + np.abs(Delay[0])
    delay_new = np.arange(Delay[0], Delay[-1], timespan/160)
    energy_new = np.linspace(Energy[0], Energy[-1], 200)
    values_new = interp2(delay_new, energy_new)

    # interpolate to momentum [a.u]
    energy_new_joules = energy_new * sc.electron_volt # joules
    energy_new_au = energy_new_joules / sc.physical_constants['atomic unit of energy'][0]  # a.u.
    momentum_new_au = np.sqrt(2 * energy_new_au)
    interp2_momentum = interpolate.interp2d(delay_new, momentum_new_au, values_new, kind='linear')

    # interpolate onto linear momentum axis
    N = len(momentum_new_au)
    momentum_linear = np.linspace(momentum_new_au[0], momentum_new_au[-1], N)
    values_lin_momentum = interp2_momentum(delay_new, momentum_linear)


    return delay_new, momentum_linear, values_lin_momentum




def create_plots():

    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(4,3)

    axes = {}

    axes['input_image'] = fig.add_subplot(gs[1,:])
    axes['input_xuv_time'] = fig.add_subplot(gs[0,2])
    axes['input_xuv'] = fig.add_subplot(gs[0,1])
    axes['input_ir'] = fig.add_subplot(gs[0,0])

    axes['generated_image'] = fig.add_subplot(gs[3,:])
    axes['predicted_xuv_time'] = fig.add_subplot(gs[2, 2])
    axes['predicted_xuv'] = fig.add_subplot(gs[2, 1])
    axes['predicted_ir'] = fig.add_subplot(gs[2, 0])

    # add time plots

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

    if actual_fields:
        xuv_actual_time = sess.run(xuv_time_domain_func, feed_dict={crab_tf2.xuv_cropped_f: actual_fields['xuv_f']})

    xuv_predicted_time = sess.run(xuv_time_domain_func, feed_dict={crab_tf2.xuv_cropped_f: predicted_fields['xuv_f']})

    # set phase angle to 0
    predicted_fields['xuv_f'] = predicted_fields['xuv_f'] * np.exp(-1j * np.angle(predicted_fields['xuv_f'][index_0_angle]))

    iteration = i+1

    # calculate loss
    loss_value = sess.run(network2.u_losses, feed_dict={network2.x: trace})
    print('iteration: {}'.format(iteration))
    print('loss_value: ', loss_value)


    if actual_fields:
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


        axes['input_xuv_time'].cla()
        axes['input_xuv_time'].text(0.0, 1.05, 'actual XUV E(t)', transform=axes['input_xuv_time'].transAxes, backgroundcolor='white')
        axes['input_xuv_time'].plot(np.real(xuv_actual_time), color='blue', alpha=0.5)
        axes['input_xuv_time'].plot(np.abs(xuv_actual_time), color='black')
        axes['input_xuv_time'].set_yticks([])
        axes['input_xuv_time'].set_xticks([])


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


    axes['predicted_xuv_time'].cla()
    axes['predicted_xuv_time'].text(0.0, 1.05, 'predicted XUV E(t)', transform=axes['predicted_xuv_time'].transAxes,
                                backgroundcolor='white')
    axes['predicted_xuv_time'].plot(np.real(xuv_predicted_time), color='blue', alpha=0.5)
    axes['predicted_xuv_time'].plot(np.abs(xuv_predicted_time), color='black')

    axes['predicted_xuv_time'].set_yticks([])
    axes['predicted_xuv_time'].set_xticks([])


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






if __name__ == "__main__":


    # run name
    # can do multiple run names for the same model
    run_name = 'experimental_retrieval'


    # copy the model to a new version to use for unsupervised learning
    modelname = 'measured_data'
    for file in glob.glob(r'./models/{}.ckpt.*'.format(modelname)):
        file_newname = file.replace(modelname, modelname+'_unsupervised')
        shutil.copy(file, file_newname)


    # get the trace
    #trace, actual_fields = get_trace(index=2, filename='attstrace_test2_processed.hdf5')

    # get trace from experimental
    _, _, trace = get_measured_trace()
    trace = trace.reshape(1, -1)
    actual_fields = None

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


        saver.save(sess, './models/{}.ckpt'.format(modelname + '_unsupervised'))




