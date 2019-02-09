import tensorflow as tf
import tf_functions
import numpy as np
import scipy.constants as sc
import tables
import shutil
import matplotlib.pyplot as plt
import os
import csv
from network3 import initialize_xuv_ir_trace_graphs, setup_neural_net, separate_xuv_ir_vec
import xuv_spectrum.spectrum
import ir_spectrum.ir_spectrum



def find_central_frequency_from_trace(trace, delay, energy, plotting=False):

    # make sure delay is even
    assert len(delay) % 2 == 0

    # find central frequency


    N = len(delay)
    print('N: ', N)
    dt = delay[-1] - delay[-2]
    df = 1 / (dt * N)
    freq_even = df * np.arange(-N / 2, N / 2)
    # plot the streaking trace and ft

    trace_f = np.fft.fftshift(np.fft.fft(np.fft.fftshift(trace, axes=1), axis=1), axes=1)

    # summation along vertical axis
    integrate = np.sum(np.abs(trace_f), axis=0)

    # find the maximum values
    f0 = find_f0(x=freq_even, y=integrate)
    if plotting:
        _, ax = plt.subplots(3, 1)
        ax[0].pcolormesh(delay, energy, trace, cmap='jet')
        ax[1].pcolormesh(freq_even, energy, np.abs(trace_f), cmap='jet')
        ax[2].plot(freq_even, integrate)



    return f0


def find_f0(x, y):

    x = np.array(x)
    y = np.array(y)

    maxvals = []

    for _ in range(3):
        max_index = np.argmax(y)
        maxvals.append(x[max_index])

        x = np.delete(x, max_index)
        y = np.delete(y, max_index)

    maxvals = np.delete(maxvals, np.argmin(np.abs(maxvals)))

    return maxvals[np.argmax(maxvals)]


def create_plot_axes():

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1)
    gs = fig.add_gridspec(3, 3)

    axes_dict = {}
    axes_dict["input_trace"] = fig.add_subplot(gs[0,:])

    axes_dict["predicted_xuv_t"] = fig.add_subplot(gs[1, 2])

    axes_dict["predicted_xuv"] = fig.add_subplot(gs[1,1])
    axes_dict["predicted_xuv_phase"] = axes_dict["predicted_xuv"].twinx()

    axes_dict["predicted_ir"] = fig.add_subplot(gs[1,0])
    axes_dict["predicted_ir_phase"] = axes_dict["predicted_ir"].twinx()

    axes_dict["generated_trace"] = fig.add_subplot(gs[2,:])

    return axes_dict




def plot_generated_trace(axes, generated_trace, xuv_coefs, ir_params, input_trace, tf_generator_graphs, sess, streak_params):



    xuv_tmat = xuv_spectrum.spectrum.tmat * sc.physical_constants['atomic unit of time'][0]*1e18 # attosecond
    xuv_fmat = xuv_spectrum.spectrum.fmat_cropped / sc.physical_constants['atomic unit of time'][0] # Hz
    ir_fmat = ir_spectrum.ir_spectrum.fmat_cropped / sc.physical_constants['atomic unit of time'][0] # Hz
    tau_values = (streak_params["tau_values"] * sc.physical_constants['atomic unit of time'][0])*1e15 # femtosecond
    k_values = streak_params["k_values"] # a.u.




    xuv_Ef = sess.run(tf_generator_graphs["xuv_E_prop"]["f_cropped"], feed_dict={tf_generator_graphs["xuv_coefs_in"]: xuv_coefs.reshape(1, -1)})
    xuv_Et = sess.run(tf_generator_graphs["xuv_E_prop"]["t"], feed_dict={tf_generator_graphs["xuv_coefs_in"]: xuv_coefs.reshape(1, -1)})

    ir_Ef = sess.run(tf_generator_graphs["ir_E_prop"]["f_cropped"],feed_dict={tf_generator_graphs["ir_values_in"]: ir_params.reshape(1, -1)})


    axes["input_trace"].pcolormesh(tau_values,k_values,input_trace, cmap="jet")
    axes["input_trace"].set_ylabel("atomic units Energy")
    axes["input_trace"].set_xlabel("fs")
    axes["input_trace"].set_title("input streaking trace")

    axes["generated_trace"].pcolormesh(tau_values,k_values,generated_trace, cmap="jet")
    axes["generated_trace"].set_xlabel("fs")
    axes["generated_trace"].set_ylabel("atomic units Energy")
    axes["generated_trace"].set_title("generated streaking trace")

    trace_actual_reshape = generated_trace.reshape(-1)
    trace_reconstructed_reshaped = input_trace.reshape(-1)
    trace_rmse = np.sqrt((1 / len(trace_actual_reshape)) * np.sum(
            (trace_reconstructed_reshaped - trace_actual_reshape) ** 2))
    axes["generated_trace"].text(0.1, 0.1, "rmse: {}".format(trace_rmse),
                                 transform=axes["generated_trace"].transAxes,
                                 backgroundcolor="white")




    axes["predicted_xuv_phase"].plot(xuv_fmat, np.unwrap(np.angle(xuv_Ef[0])), color='green')
    axes["predicted_xuv"].plot(xuv_fmat, np.abs(xuv_Ef[0])**2, label="Intensity", color="black")
    axes["predicted_xuv"].set_xlabel("Hz")
    axes["predicted_xuv"].legend()


    axes["predicted_xuv_t"].plot(xuv_tmat, np.real(xuv_Et[0]), color="blue", label="xuv E(t)")
    axes["predicted_xuv_t"].set_xlabel("attoseconds")
    axes["predicted_xuv_t"].legend()

    axes["predicted_ir"].plot(ir_fmat, np.real(ir_Ef[0]), color="blue")
    axes["predicted_ir"].plot(ir_fmat, np.imag(ir_Ef[0]), color="red")
    axes["predicted_ir"].set_xlabel("Hz")



def get_measured_trace():



    filepath = './measured_trace/sample2/MSheet1_1.csv'
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        matrix = np.array(list(reader))

        Energy = matrix[1:, 0].astype('float') # eV
        Delay = matrix[0, 1:].astype('float') # fs
        Values = matrix[1:, 1:].astype('float')

    #print(Delay)
    # print('len(Energy): ', len(Energy))
    # print('Energy: ', Energy)


    # construct frequency axis with even number for fourier transform
    values_even = Values[:, :-1]
    Delay_even = Delay[:-1]
    Delay_even = Delay_even * 1e-15  # convert to seconds
    Dtau = Delay_even[-1] - Delay_even[-2]
    # print('Delay: ', Delay)
    # print('Delay_even: ', Delay_even)
    # print('np.shape(values_even): ', np.shape(values_even))
    # print('len(values_even.reshape(-1))', len(values_even.reshape(-1)))
    # print('Dtau: ', Dtau)
    # print('Delay max', Delay_even[-1])
    # print('N: ', len(Delay_even))
    # print('Energy: ', len(Energy))
    f0 = find_central_frequency_from_trace(trace=values_even, delay=Delay_even, energy=Energy)
    # print(f0)  # in seconds
    lam0 = sc.c / f0
    # print('f0 a.u.: ', f0 * sc.physical_constants['atomic unit of time'][0])  # convert f0 to atomic unit
    # print('lam0: ', lam0)


    # normalize values

    #exit(0)
    return Delay_even, Energy, values_even



if __name__ == "__main__":


    tf_generator_graphs, streak_params, xuv_phase_coefs = initialize_xuv_ir_trace_graphs()

    nn_nodes = setup_neural_net(streak_params, xuv_phase_coefs)

    axes = create_plot_axes()


    _, _, trace = get_measured_trace()

    modelname = 'run3'


    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './models/{}.ckpt'.format(modelname))

        predicted_fields = sess.run(nn_nodes["y_pred"], feed_dict={nn_nodes["x"]: trace.reshape(1, -1)})

        xuv_coefs, ir_params = separate_xuv_ir_vec(predicted_fields[0])



        generated_trace = sess.run(tf_generator_graphs["image"],
                                   feed_dict={tf_generator_graphs["xuv_coefs_in"]: xuv_coefs.reshape(1, -1),
                                              tf_generator_graphs["ir_values_in"]: ir_params.reshape(1, -1)})

        print(np.shape(generated_trace))

        plot_generated_trace(axes=axes, generated_trace=generated_trace, xuv_coefs=xuv_coefs,
                             ir_params=ir_params, input_trace=trace, tf_generator_graphs=tf_generator_graphs,
                             sess=sess, streak_params=streak_params)

        plt.show()


