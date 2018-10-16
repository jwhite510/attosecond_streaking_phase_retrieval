import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy import interpolate

# load the model functions
import importlib
network_name = 'reg_conv_net_1'
model = importlib.import_module('models.network_{}_'.format(network_name))

# load the model data
import tensorflow as tf



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


with tf.Session() as sess:
    # restore checkpoint
    saver = tf.train.Saver()
    saver.restore(sess, './models/{}.ckpt'.format(network_name))


