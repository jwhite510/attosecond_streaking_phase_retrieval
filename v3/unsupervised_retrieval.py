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
from xuv_spectrum import spectrum
from ir_spectrum import ir_spectrum
import glob
import pickle



def update_plots(sess, nn_nodes, axes, measured_trace, i, run_name, streak_params, retrieval):

    if retrieval == "normal":

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
        # .....CALCULATE RMSE........
        # ...........................
        # calculate rmse
        trace_rmse = np.sqrt(
            (1 / len(measured_trace.reshape(-1))) * np.sum(
                (measured_trace.reshape(-1) - reconstruced.reshape(-1)) ** 2))




        # ...........................
        # ........PLOTTING...........
        # ...........................
        # input trace
        axes["input_trace"].pcolormesh(streak_params["tau_values"], streak_params["k_values"], measured_trace, cmap='jet')
        axes["input_trace"].text(0.0, 1.0, "actual_trace", backgroundcolor="white",
                                                         transform=axes["input_trace"].transAxes)
        axes["input_trace"].text(0.5, 1.0, "Unsupervised Learning", backgroundcolor="white",
                                 transform=axes["input_trace"].transAxes)

        # generated trace
        axes["generated_trace"].pcolormesh(streak_params["tau_values"], streak_params["k_values"], reconstruced, cmap='jet')
        axes["generated_trace"].text(0.1, 0.1, "RMSE: {}".format(str(np.round(trace_rmse, 3))),
                                     transform=axes["generated_trace"].transAxes,
                                     backgroundcolor="white")
        axes["generated_trace"].text(0.0, 1.0, "generated_trace", backgroundcolor="white",
                                                            transform=axes["generated_trace"].transAxes)
        # xuv predicted
        # xuv t
        axes["predicted_xuv_t"].plot(spectrum.tmat, np.abs(xuv_t)**2, color="black")
        # xuv f
        # axes["predicted_xuv"].plot(np.real(xuv_f), color="blue")
        # axes["predicted_xuv"].plot(np.imag(xuv_f), color="red")
        axes["predicted_xuv"].plot(spectrum.fmat_cropped, np.abs(xuv_f)**2, color="black")
        axes["predicted_xuv_phase"].text(0.0, 1.1, "predicted_xuv", backgroundcolor="white",
                                                                transform=axes["predicted_xuv_phase"].transAxes)

        axes["predicted_xuv_phase"].plot(spectrum.fmat_cropped, np.unwrap(np.angle(xuv_f)), color="green")

        # ir predicted
        axes["predicted_ir"].plot(ir_spectrum.fmat_cropped, np.abs(ir_f)**2, color="black")
        axes["predicted_ir_phase"].plot(ir_spectrum.fmat_cropped, np.unwrap(np.angle(ir_f)), color="green")

        # save files
        dir = "./unsupervised_retrieval/" + run_name + "/"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        plt.savefig(dir + str(i) + ".png")
        with open("./unsupervised_retrieval/"+run_name+"/u_fields.p", "wb") as file:


            predicted_fields = {}
            predicted_fields["ir_f"] = ir_f
            predicted_fields["xuv_f"] = xuv_f
            predicted_fields["xuv_t"] = xuv_t

            save_files = {}
            save_files["predicted_fields"] = predicted_fields
            save_files["measured_trace"] = measured_trace
            save_files["reconstruced"] = reconstruced
            save_files["iteration"] = i
            pickle.dump(save_files,file)


        plt.pause(0.00001)

    elif retrieval == "proof":

        feed_dict = {nn_nodes["general"]["x_in"]: measured_trace.reshape(1, -1)}
        input_proof = sess.run(nn_nodes["unsupervised"]["proof"]["input_image_proof"]["proof"], feed_dict=feed_dict)
        reconstruced_proof = sess.run(nn_nodes["unsupervised"]["proof"]["reconstructed_proof"]["proof"], feed_dict=feed_dict)
        ir_f = sess.run(nn_nodes["general"]["phase_net_output"]["ir_E_prop"]["f_cropped"], feed_dict=feed_dict)[0]
        xuv_f = sess.run(nn_nodes["general"]["phase_net_output"]["xuv_E_prop"]["f_cropped"], feed_dict=feed_dict)[0]
        xuv_t = sess.run(nn_nodes["general"]["phase_net_output"]["xuv_E_prop"]["t"], feed_dict=feed_dict)[0]

        # ...........................
        # ........CLEAR AXES.........
        # ...........................
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
        # .....CALCULATE RMSE........
        # ...........................
        # calculate rmse
        proof_rmse = np.sqrt(
            (1 / len(input_proof.reshape(-1))) * np.sum(
                (input_proof.reshape(-1) - reconstruced_proof.reshape(-1)) ** 2))

        # ...........................
        # ........PLOTTING...........
        # ...........................
        # input trace
        axes["input_trace"].pcolormesh(streak_params["tau_values"], streak_params["k_values"], input_proof,
                                       cmap='jet')
        axes["input_trace"].text(0.0, 1.0, "actual_trace", backgroundcolor="white",
                                 transform=axes["input_trace"].transAxes)
        axes["input_trace"].text(0.5, 1.0, "Unsupervised Learning", backgroundcolor="white",
                                 transform=axes["input_trace"].transAxes)

        # generated trace
        axes["generated_trace"].pcolormesh(streak_params["tau_values"], streak_params["k_values"], reconstruced_proof,
                                           cmap='jet')
        axes["generated_trace"].text(0.1, 0.1, "RMSE: {}".format(str(np.round(proof_rmse, 3))),
                                     transform=axes["generated_trace"].transAxes,
                                     backgroundcolor="white")
        axes["generated_trace"].text(0.0, 1.0, "generated_trace", backgroundcolor="white",
                                     transform=axes["generated_trace"].transAxes)
        # xuv predicted
        # xuv t
        axes["predicted_xuv_t"].plot(spectrum.tmat, np.abs(xuv_t) ** 2, color="black")
        # xuv f
        # axes["predicted_xuv"].plot(np.real(xuv_f), color="blue")
        # axes["predicted_xuv"].plot(np.imag(xuv_f), color="red")
        axes["predicted_xuv"].plot(spectrum.fmat_cropped, np.abs(xuv_f) ** 2, color="black")
        axes["predicted_xuv_phase"].text(0.0, 1.1, "predicted_xuv", backgroundcolor="white",
                                         transform=axes["predicted_xuv_phase"].transAxes)

        axes["predicted_xuv_phase"].plot(spectrum.fmat_cropped, np.unwrap(np.angle(xuv_f)), color="green")

        # ir predicted
        axes["predicted_ir"].plot(ir_spectrum.fmat_cropped, np.abs(ir_f) ** 2, color="black")
        axes["predicted_ir_phase"].plot(ir_spectrum.fmat_cropped, np.unwrap(np.angle(ir_f)), color="green")

        # save files
        dir = "./unsupervised_retrieval/" + run_name + "/"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        plt.savefig(dir + str(i) + ".png")
        with open("./unsupervised_retrieval/"+run_name+"/u_fields_proof.p", "wb") as file:

            predicted_fields = {}
            predicted_fields["ir_f"] = ir_f
            predicted_fields["xuv_f"] = xuv_f
            predicted_fields["xuv_t"] = xuv_t

            save_files = {}
            save_files["predicted_fields"] = predicted_fields
            save_files["input_proof"] = input_proof
            save_files["reconstruced_proof"] = reconstruced_proof
            save_files["iteration"] = i
            pickle.dump(save_files, file)

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

    run_name = "proof1"

    #===================
    #==Retrieval Type===
    #===================
    # retrieval = "normal"
    retrieval = "proof"


    # copy the model to a new version to use for unsupervised learning
    modelname = "test1_phasecurve_proof1"
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

    # init data object
    get_data = network3.GetData(batch_size=10)


    axes = create_plot_axes()







    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './models/{}.ckpt'.format(modelname+'_unsupervised'))


        # get the initial output
        reconstruced = sess.run(nn_nodes["general"]["reconstructed_trace"],
                                feed_dict={nn_nodes["general"]["x_in"]: measured_trace.reshape(1, -1)})


        plt.ion()
        for i in range(999999):


            if i % 100 == 0:
                print(i)
                # get MSE between traces
                summ = sess.run(unsupervised_mse_tb,
                                feed_dict={nn_nodes["general"]["x_in"]: measured_trace.reshape(1, -1)})
                writer.add_summary(summ, global_step=i + 1)
                writer.flush()

                # update plots
                update_plots(sess=sess, nn_nodes=nn_nodes, axes=axes, measured_trace=measured_trace, i=i+1, run_name=run_name,
                             streak_params=streak_params, retrieval=retrieval)



            # train neural network
            #========================
            #=======logairthmic======
            #========================
            #sess.run(nn_nodes["unsupervised"]["unsupervised_train_log"],
            #         feed_dict={
            #             nn_nodes["unsupervised"]["u_LR"]: 0.00001,
            #             nn_nodes["unsupervised"]["x_in"]: measured_trace.reshape(1, -1),
            #             nn_nodes["unsupervised"]["u_base"]: 10.0,
            #             nn_nodes["unsupervised"]["u_translate"]: 1.0
            #         })

            # ========================
            # =========proof==========
            # ========================
            sess.run(nn_nodes["unsupervised"]["proof"]["proof_unsupervised_train"],
                     feed_dict={
                         nn_nodes["unsupervised"]["proof"]["u_LR"]: 0.00001,
                         nn_nodes["unsupervised"]["proof"]["x_in"]: measured_trace.reshape(1, -1),
                     })

            # ========================
            # =========supervised=====
            # ========================
            # retrieve data
            #if get_data.batch_index >= get_data.samples:
            #    get_data.batch_index = 0
            #batch_x, batch_y = get_data.next_batch()
            #sess.run(nn_nodes["supervised"]["phase_network_train_coefs_params"],
            #         feed_dict={nn_nodes["supervised"]["x_in"]: batch_x,
            #                    nn_nodes["supervised"]["actual_coefs_params"]: batch_y,
            #                    nn_nodes["general"]["hold_prob"]: 0.8,
            #                    nn_nodes["supervised"]["s_LR"]: 0.0001})





