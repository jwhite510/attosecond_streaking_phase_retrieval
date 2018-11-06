import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy import interpolate
from generate_data import XUV_Field_rand_phase
import importlib
modelname = 'reg_conv_net_11_5_18_linmomentum'
# model = importlib.import_module('models.network_{}'.format(modelname))
from models.network_reg_conv_net_11_5_18_linmomentum import *
import tensorflow as tf
import scipy.constants as sc




def plot_predictions(x_in, y_in, axis, fig, set, modelname, epoch):

    mses = []
    predictions = sess.run(y_pred, feed_dict={x: x_in,
                                              y_true: y_in})

    # for ax, index in zip([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]):
    for ax, index in zip([0, 1, 2], [0, 1, 2]):

        mse = sess.run(loss, feed_dict={x: x_in[index].reshape(1, -1),
                                        y_true: y_in[index].reshape(1, -1)})
        mses.append(mse)

        # plot  actual trace
        axis[0][ax].pcolormesh(x_in[index].reshape(len(generate_proof_traces.p_vec),
                                                   len(generate_proof_traces.tauvec)), cmap='jet')

        # set the number of points to crop when plotting the phase to redice the noise
        crop_phase_right = 15
        crop_phase_left = 20

        # plot E(t) retrieved
        axis[1][ax].cla()
        complex_field = predictions[index, :64] + 1j * predictions[index, 64:]
        axis[1][ax].plot(electronvolts, np.abs(complex_field), color="black")
        axtwin = axis[1][ax].twinx()
        axtwin.plot(electronvolts[crop_phase_left:-crop_phase_right], np.unwrap(np.angle(complex_field))[crop_phase_left:-crop_phase_right], color="green")
        axis[1][ax].text(0.1, 1, "prediction [" + set + " set]", transform=axis[1][ax].transAxes,
                         backgroundcolor='white')

        # plot the error
        axis[1][ax].text(0.1, 1.1, "MSE: " + str(mse),
                         transform=axis[1][ax].transAxes, backgroundcolor='white')


        # plot E(t) actual
        axis[2][ax].cla()
        complex_field = y_in[index, :64] + 1j * y_in[index, 64:]
        axis[2][ax].plot(electronvolts, np.abs(complex_field), color="black")
        axtwin = axis[2][ax].twinx()
        axtwin.plot(electronvolts[crop_phase_left:-crop_phase_right], np.unwrap(np.angle(complex_field))[crop_phase_left:-crop_phase_right], color="green")
        axis[2][ax].text(0.1, 1, "actual [" + set + " set]", transform=axis[2][ax].transAxes,
                         backgroundcolor='white')

        # axis[0][ax].set_xticks([])
        # axis[0][ax].set_yticks([])
        # axis[1][ax].set_xticks([])
        # axis[1][ax].set_yticks([])
        # axis[2][ax].set_xticks([])
        # axis[2][ax].set_yticks([])


    print("mses: ", mses)
    print("avg : ", (1 / len(mses)) * np.sum(np.array(mses)))

    # save image
    dir = "/home/zom/PythonProjects/attosecond_streaking_phase_retrieval/nnpictures/" + modelname + "/" + set + "/"
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig.savefig(dir + str(epoch) + ".png")



def retrieve_pulse(filepath, plotting=False):
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        matrix = np.array(list(reader))

        Energy = matrix[4:, 0].astype('float')
        Delay = matrix[2, 2:].astype('float')
        Values = matrix[4:, 2:].astype('float')

    # map the function onto a
    interp2 = interpolate.interp2d(Delay, Energy, Values, kind='linear')

    delay_new = np.linspace(Delay[0], Delay[-1], 176)
    energy_new = np.linspace(Energy[0], Energy[-1], 200)

    values_new = interp2(delay_new, energy_new)

    if plotting:

        fig = plt.figure()
        gs = fig.add_gridspec(2, 2)

        ax = fig.add_subplot(gs[0, :])
        ax.pcolormesh(Delay, Energy, Values, cmap='jet')
        ax.set_xlabel('fs')
        ax.set_ylabel('eV')
        ax.text(0.1, 0.9, 'original', transform=ax.transAxes, backgroundcolor='white')

        ax = fig.add_subplot(gs[1, :])
        ax.pcolormesh(delay_new, energy_new, values_new, cmap='jet')
        ax.set_xlabel('fs')
        ax.set_ylabel('eV')
        ax.text(0.1, 0.9, 'interpolated', transform=ax.transAxes, backgroundcolor='white')

        plt.show()

    return delay_new, energy_new, values_new


# retrieve the experimental data
delay, energy, trace = retrieve_pulse(filepath='./experimental_data/53asstreakingdata.csv', plotting=False)


# retrieve f vector
xuv_test = XUV_Field_rand_phase(phase_amplitude=0, phase_nodes=100, plot=False)
fmat = xuv_test.f_cropped_cropped # a.u.
fmat_hz = fmat / sc.physical_constants['atomic unit of time'][0]
fmat_joules = sc.h * fmat_hz # joules
electronvolts = 1 / (sc.elementary_charge) * fmat_joules



#initialize the plot
fig2, ax2 = plt.subplots(3, 3, figsize=(14, 8))
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05,
                        wspace=0.1, hspace=0.1)

with tf.Session() as sess:
    # restore checkpoint
    saver = tf.train.Saver()
    print('restoring ', './models/{}.ckpt'.format(modelname))
    saver.restore(sess, './models/{}.ckpt'.format(modelname))
    get_data = GetData(batch_size=10)

    # get data and evaluate
    batch_x_test, batch_y_test = get_data.evaluate_on_test_data()
    plot_predictions(x_in=batch_x_test, y_in=batch_y_test, axis=ax2, fig=fig2,
                     set="test", modelname=modelname, epoch=0)

    plt.ioff()
    plt.show()


