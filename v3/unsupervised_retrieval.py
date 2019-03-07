import tensorflow as tf
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import tables
import shutil
import matplotlib.pyplot as plt
import os
import csv
# from network3 import initialize_xuv_ir_trace_graphs, setup_neural_net, separate_xuv_ir_vec
import network3
import xuv_spectrum.spectrum
import ir_spectrum.ir_spectrum
import glob



def update_plots(sess, nn_nodes, axes, measured_trace):



    feed_dict = {nn_nodes["general"]["x_in"]: measured_trace.reshape(1, -1)}
    reconstruced = sess.run(nn_nodes["general"]["reconstructed_trace"],feed_dict=feed_dict)
    ir_f = sess.run(nn_nodes["general"]["phase_net_output"]["ir_E_prop"]["f_cropped"],feed_dict=feed_dict)[0]
    xuv_f = sess.run(nn_nodes["general"]["phase_net_output"]["xuv_E_prop"]["f_cropped"],feed_dict=feed_dict)[0]
    xuv_t = sess.run(nn_nodes["general"]["phase_net_output"]["xuv_E_prop"]["t"],feed_dict=feed_dict)[0]

    #...........................
    #........CLEAR AXES.........
    #...........................
    # input trace
    axes["input_trace"].cla()
    # xuv predicted
    axes["predicted_xuv_t"].cla()
    axes["predicted_xuv"].cla()
    axes["predicted_xuv_phase"].cla()
    # predicted ir
    axes["predicted_ir"].cla()
    axes["predicted_ir_phase"].cla()
    # generated trace
    axes["generated_trace"].cla()


    # ...........................
    # ........PLOTTING...........
    # ...........................
    # input trace
    axes["input_trace"].pcolormesh(measured_trace, cmap='jet')

    # generated trace
    axes["generated_trace"].pcolormesh(reconstruced, cmap='jet')

    # xuv predicted
    axes["predicted_xuv_t"].plot(np.abs(xuv_t)**2, color="black")
    axes["predicted_xuv"].plot(np.real(xuv_f), color="blue")
    axes["predicted_xuv"].plot(np.imag(xuv_f), color="red")
    axes["predicted_xuv"].plot(np.abs(xuv_f), color="black")
    axes["predicted_xuv_phase"].plot(np.unwrap(np.angle(xuv_f)), color="green")

    # ir predicted
    axes["predicted_ir"].plot(np.abs(ir_f)**2, color="black")
    axes["predicted_ir_phase"].plot(np.unwrap(np.angle(ir_f)), color="green")


    plt.pause(0.00001)







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
    # Dtau = Delay_even[-1] - Delay_even[-2]
    # print('Delay: ', Delay)
    # print('Delay_even: ', Delay_even)
    # print('np.shape(values_even): ', np.shape(values_even))
    # print('len(values_even.reshape(-1))', len(values_even.reshape(-1)))
    # print('Dtau: ', Dtau)
    # print('Delay max', Delay_even[-1])
    # print('N: ', len(Delay_even))
    # print('Energy: ', len(Energy))
    # f0 = find_central_frequency_from_trace(trace=values_even, delay=Delay_even, energy=Energy)
    # print(f0)  # in seconds
    # lam0 = sc.c / f0
    # print('f0 a.u.: ', f0 * sc.physical_constants['atomic unit of time'][0])  # convert f0 to atomic unit
    # print('lam0: ', lam0)


    # normalize values

    #exit(0)
    return Delay_even, Energy, values_even





if __name__ == "__main__":

    run_name = "run3"

    # copy the model to a new version to use for unsupervised learning
    modelname = "test1_abs_I"
    for file in glob.glob(r'./models/{}.ckpt.*'.format(modelname)):
        file_newname = file.replace(modelname, modelname+'_unsupervised')
        shutil.copy(file, file_newname)

    # get the measured trace
    _, _, measured_trace = get_measured_trace()


    # initialize xuv, IR, and trace graphs
    tf_generator_graphs, streak_params = network3.initialize_xuv_ir_trace_graphs()

    # build neural net graph
    nn_nodes = network3.setup_neural_net(streak_params)

    # create mse measurer
    writer = tf.summary.FileWriter("./tensorboard_graph_u/" + run_name)
    unsupervised_mse_tb = tf.summary.scalar("trace_mse", nn_nodes["unsupervised"]["unsupervised_learning_loss"])


    axes = create_plot_axes()




    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './models/{}.ckpt'.format(modelname+'_unsupervised'))


        # get the initial output
        reconstruced = sess.run(nn_nodes["general"]["reconstructed_trace"],
                                feed_dict={nn_nodes["general"]["x_in"]: measured_trace.reshape(1, -1)})


        plt.ion()
        for i in range(9999):


            if i % 10 == 0:
                print(i)
                # get MSE between traces
                summ = sess.run(unsupervised_mse_tb,
                                feed_dict={nn_nodes["general"]["x_in"]: measured_trace.reshape(1, -1)})
                writer.add_summary(summ, global_step=i + 1)
                writer.flush()

                # update plots
                update_plots(sess=sess, nn_nodes=nn_nodes, axes=axes, measured_trace=measured_trace)

            # train neural network
            sess.run(nn_nodes["unsupervised"]["unsupervised_train"],
                     feed_dict={
                         nn_nodes["unsupervised"]["u_LR"]: 0.0001,
                         nn_nodes["unsupervised"]["x_in"]: measured_trace.reshape(1, -1)
                     })


