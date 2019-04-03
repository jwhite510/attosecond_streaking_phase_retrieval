import tensorflow as tf
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import tables
import shutil
import os
import csv
import network3
from xuv_spectrum import spectrum
from phase_parameters import params
from ir_spectrum import ir_spectrum
import glob
import pickle
import tf_functions
import measured_trace.get_trace as measured_trace

def apply_noise(trace, counts):

    discrete_trace = np.round(trace * counts)
    noise = np.random.poisson(lam=discrete_trace) - discrete_trace
    noisy_trace = discrete_trace + noise
    noisy_trace_normalized = noisy_trace / np.max(noisy_trace)
    return noisy_trace_normalized


def get_fake_measured_trace(counts, plotting, run_name=None):

    # initialize XUV generator
    xuv_coefs_in = tf.placeholder(tf.float32, shape=[None, params.xuv_phase_coefs])
    xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_in)

    # initialize IR generator
    ir_values_in = tf.placeholder(tf.float32, shape=[None, 4])
    ir_E_prop = tf_functions.ir_from_params(ir_values_in)

    # construct streaking image
    image = tf_functions.streaking_trace(xuv_cropped_f_in=xuv_E_prop["f_cropped"][0],
                                         ir_cropped_f_in=ir_E_prop["f_cropped"][0])
    proof_trace = tf_functions.proof_trace(image)
    autocorelation = tf_functions.autocorrelate(image)

    tf_graphs = {}
    tf_graphs["xuv_coefs_in"] = xuv_coefs_in
    tf_graphs["ir_values_in"] = ir_values_in
    tf_graphs["image"] = image
    tf_graphs["proof_trace"] = proof_trace
    tf_graphs["autocorelation"] = autocorelation

    xuv_input = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
    ir_input = np.array([[0.0, 0.0, 0.0, 0.0]])

    with tf.Session() as sess:
        feed_dict = {tf_graphs["xuv_coefs_in"]: xuv_input, tf_graphs["ir_values_in"]: ir_input}
        trace = sess.run(tf_graphs["image"], feed_dict=feed_dict)
        trace_proof = sess.run(tf_graphs["proof_trace"]["proof"], feed_dict=feed_dict)
        trace_autocorrelation = sess.run(tf_graphs["autocorelation"], feed_dict=feed_dict)
        xuv_t = sess.run(xuv_E_prop["t"], feed_dict=feed_dict)[0]
        xuv_f = sess.run(xuv_E_prop["f_cropped"], feed_dict=feed_dict)[0]
        ir_f = sess.run(ir_E_prop["f_cropped"], feed_dict=feed_dict)[0]
        # construct proof and autocorrelate from non-noise trace

    # generated from noise free trace
    traces = {}
    traces["trace"] = trace
    traces["autocorrelation"] = trace_autocorrelation
    traces["proof"] = trace_proof

    # construct noisy trace and other traces from noisy trace
    noisy_trace = apply_noise(trace, counts)
    # generate proof and autocorrelation from noisy trace
    noisy_autocorrelation = tf_functions.autocorrelate(tf.constant(noisy_trace, dtype=tf.float32))
    noisy_proof = tf_functions.proof_trace(tf.constant(noisy_trace, dtype=tf.float32))
    with tf.Session() as sess:
        noisy_autocorrelation_trace = sess.run(noisy_autocorrelation)
        noisy_proof_trace = sess.run(noisy_proof["proof"])

    # generated from noisy trace
    noise_traces = {}
    noise_traces["trace"] = noisy_trace
    noise_traces["autocorrelation"] = noisy_autocorrelation_trace
    noise_traces["proof"] = noisy_proof_trace

    tf.reset_default_graph()

    if plotting:
        axes = create_plot_axes()
        plot_images_fields(axes=axes, trace_meas=noisy_trace, trace_reconstructed=trace,
                           xuv_f=xuv_f, xuv_t=xuv_t, ir_f=ir_f, i=None, trace_yaxis=params.K,
                           run_name=None, true_fields=True)

        # save files
        dir = "./unsupervised_retrieval/" + run_name + "/"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        plt.savefig(dir+"actual_fields" + str(counts) + ".png")

    return noisy_trace


def calc_fwhm(tmat, I_t):

    half_max = np.max(I_t)/2
    index1 = 0
    index2 = len(I_t) - 1

    while I_t[index1] < half_max:
        index1 += 1
    while I_t[index2] < half_max:
        index2 -= 1

    t1 = tmat[index1]
    t2 = tmat[index2]
    fwhm = t2 - t1
    return fwhm, t1, t2, half_max


def plot_images_fields(axes, traces_meas, traces_reconstructed, xuv_f, xuv_t, ir_f, i, trace_yaxis,
                       run_name, true_fields=False):

    trace_meas = traces_meas["measured_trace"]
    trace_reconstructed = traces_reconstructed["reconstructed"]

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
    # calculate rmse for each trace

    #==================================
    #==================================
    #==================================
    #==================================

    # for measured, reconstructed in zip(traces_meas, traces_reconstructed):
    #     # calculate the rmse for each trace
    #     trace_rmse = np.sqrt((1 / len(trace_meas.reshape(-1))) * np.sum(
    #                         (trace_meas.reshape(-1) - trace_reconstructed.reshape(-1)) ** 2))


    # calculate the rmse for each trace
    trace_rmse = np.sqrt((1 / len(trace_meas.reshape(-1))) * np.sum(
        (trace_meas.reshape(-1) - trace_reconstructed.reshape(-1)) ** 2))
    # ...........................
    # ........PLOTTING...........
    # ...........................


    # input trace
    axes["input_trace"].pcolormesh(params.delay_values_fs, trace_yaxis, trace_meas, cmap='jet')
    axes["input_trace"].set_xlabel(r"$\tau$ Delay [fs]")

    if true_fields:
        axes["input_trace"].text(0.0, 1.0, "noisy_trace", backgroundcolor="white",
                                 transform=axes["input_trace"].transAxes)
        axes["input_trace"].text(0.5, 1.1, "Actual Fields", backgroundcolor="yellow", ha="center",
                                 transform=axes["input_trace"].transAxes)
    else:
        axes["input_trace"].text(0.0, 1.0, "actual_trace", backgroundcolor="white",
                                 transform=axes["input_trace"].transAxes)
        axes["input_trace"].text(0.5, 1.1, "Unsupervised Learning", backgroundcolor="white", ha="center",
                                 transform=axes["input_trace"].transAxes)


    # generated trace
    axes["generated_trace"].pcolormesh(params.delay_values_fs, trace_yaxis, trace_reconstructed, cmap='jet')
    axes["generated_trace"].text(0.1, 0.1, "RMSE: {}".format(str(np.round(trace_rmse, 3))),
                                 transform=axes["generated_trace"].transAxes,
                                 backgroundcolor="white")
    axes["generated_trace"].set_xlabel(r"$\tau$ Delay [fs]")
    if true_fields:
        axes["generated_trace"].text(0.0, 1.0, "actual_trace", backgroundcolor="white",
                                     transform=axes["generated_trace"].transAxes)
    else:
        axes["generated_trace"].text(0.0, 1.0, "generated_trace", backgroundcolor="white",
                                     transform=axes["generated_trace"].transAxes)

    # xuv predicted
    # xuv t
    tmat_as = spectrum.tmat * sc.physical_constants['atomic unit of time'][0] * 1e18
    I_t = np.abs(xuv_t) ** 2
    axes["predicted_xuv_t"].plot(tmat_as, I_t, color="black")
    #calculate FWHM
    fwhm, t1, t2, half_max = calc_fwhm(tmat=tmat_as, I_t=I_t)
    axes["predicted_xuv_t"].text(1.0, 0.9, "FWHM:\n %.2f [as]" % fwhm, color="red", backgroundcolor="white", ha="center",
                                 transform=axes["predicted_xuv_t"].transAxes)
    #plot FWHM
    axes["predicted_xuv_t"].plot([t1, t2], [half_max, half_max], color="red", linewidth=2.0)
    axes["predicted_xuv_t"].set_yticks([])
    axes["predicted_xuv_t"].set_xlabel("time [as]")

    if true_fields:
        axes["predicted_xuv_t"].text(0.0, 1.1, "actual XUV $I(t)$", backgroundcolor="white",
                                     transform=axes["predicted_xuv_t"].transAxes)
    else:
        axes["predicted_xuv_t"].text(0.0, 1.1, "predicted XUV $I(t)$", backgroundcolor="white",
                                         transform=axes["predicted_xuv_t"].transAxes)



    # xuv f
    fmat_hz = spectrum.fmat_cropped/sc.physical_constants['atomic unit of time'][0]*1e-17
    axes["predicted_xuv"].plot(fmat_hz, np.abs(xuv_f) ** 2, color="black")
    axes["predicted_xuv"].set_yticks([])
    axes["predicted_xuv"].set_xlabel("Frequency [$10^{17}$Hz]")

    if true_fields:
        axes["predicted_xuv_phase"].text(0.0, 1.1, "actual XUV spectrum", backgroundcolor="white",
                                         transform=axes["predicted_xuv_phase"].transAxes)
    else:
        axes["predicted_xuv_phase"].text(0.0, 1.1, "predicted XUV spectrum", backgroundcolor="white",
                                         transform=axes["predicted_xuv_phase"].transAxes)

    axes["predicted_xuv_phase"].tick_params(axis='y', colors='green')
    axes["predicted_xuv_phase"].plot(fmat_hz, np.unwrap(np.angle(xuv_f)), color="green")



    # ir predicted
    fmat_ir_hz = ir_spectrum.fmat_cropped/sc.physical_constants['atomic unit of time'][0]*1e-14
    axes["predicted_ir"].plot(fmat_ir_hz, np.abs(ir_f) ** 2, color="black")
    axes["predicted_ir"].set_yticks([])
    axes["predicted_ir"].set_xlabel("Frequency [$10^{14}$Hz]")
    axes["predicted_ir_phase"].plot(fmat_ir_hz, np.unwrap(np.angle(ir_f)), color="green")
    axes["predicted_ir_phase"].tick_params(axis='y', colors='green')
    if true_fields:
        axes["predicted_ir_phase"].text(0.0, 1.1, "actual IR spectrum", backgroundcolor="white",
                                        transform=axes["predicted_ir_phase"].transAxes)
    else:
        axes["predicted_ir_phase"].text(0.0, 1.1, "predicted IR spectrum", backgroundcolor="white",
                                        transform=axes["predicted_ir_phase"].transAxes)

    # if true fields arent passed as an input
    # retrieval is running, so save images and fields
    if not true_fields:
        # save files
        dir = "./unsupervised_retrieval/" + run_name + "/"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        plt.savefig(dir + str(i) + ".png")
        with open("./unsupervised_retrieval/" + run_name + "/u_fields.p", "wb") as file:
            predicted_fields = {}
            predicted_fields["ir_f"] = ir_f
            predicted_fields["xuv_f"] = xuv_f
            predicted_fields["xuv_t"] = xuv_t

            save_files = {}
            save_files["predicted_fields"] = predicted_fields
            save_files["trace_meas"] = trace_meas
            save_files["trace_reconstructed"] = trace_reconstructed
            save_files["i"] = i
            pickle.dump(save_files, file)


def show_proof_calculation(trace, sess, nn_nodes):

    feed_dict = {nn_nodes["general"]["x_in"]: trace.reshape(1, -1)}
    out = sess.run(nn_nodes["unsupervised"]["proof"]["input_image_proof"],
                    feed_dict=feed_dict)

    fig = plt.figure()
    gs = fig.add_gridspec(4,3)

    # plot the input trace
    ax = fig.add_subplot(gs[0,:])
    ax.pcolormesh(trace)

    # plot ft of the trace
    ax = fig.add_subplot(gs[1, :])
    ax.pcolormesh(np.abs(out["freq"]))

    # plot the summation
    ax = fig.add_subplot(gs[2,:])
    ax.plot(out["summationf"])


    # mark the indexes
    ax.plot([out["w1_indexes"][0], out["w1_indexes"][0]], [np.max(out["summationf"]), 0], color="red")
    ax.plot([out["w1_indexes"][1], out["w1_indexes"][1]], [np.max(out["summationf"]), 0], color="red")

    # plot the proof trace
    ax = fig.add_subplot(gs[3, :])
    ax.pcolormesh(out["proof"])


def update_plots(sess, nn_nodes, axes, measured_trace, i, retrieval, run_name):
     
    feed_dict = {nn_nodes["general"]["x_in"]: measured_trace.reshape(1, -1)}
     
    ir_f = sess.run(nn_nodes["general"]["phase_net_output"]["ir_E_prop"]["f_cropped"],feed_dict=feed_dict)[0]
    xuv_f = sess.run(nn_nodes["general"]["phase_net_output"]["xuv_E_prop"]["f_cropped"],feed_dict=feed_dict)[0]
    xuv_t = sess.run(nn_nodes["general"]["phase_net_output"]["xuv_E_prop"]["t"],feed_dict=feed_dict)[0]

    #================================================
    #==================INPUT TRACES==================
    #================================================

    # calculate INPUT Autocorrelation from input image
    input_auto = sess.run(nn_nodes["unsupervised"]["autocorrelate"]["input_image_autocorrelate"], feed_dict=feed_dict)

    # calculate INPUT PROOF trace from input image
    input_proof = sess.run(nn_nodes["unsupervised"]["proof"]["input_image_proof"]["proof"], feed_dict=feed_dict)

    #================================================
    #==================MEASURED TRACES===============
    #================================================
    
    # reconstructed regular trace from input image
    reconstructed = sess.run(nn_nodes["general"]["reconstructed_trace"],feed_dict=feed_dict)

    # calculate reconstructed proof trace
    reconstructed_proof = sess.run(nn_nodes["unsupervised"]["proof"]["reconstructed_proof"]["proof"], feed_dict=feed_dict)

    # calculate reconstructed autocorrelation trace
    reconstruced_auto = sess.run(nn_nodes["unsupervised"]["autocorrelate"]["reconstructed_autocorrelate"], feed_dict=feed_dict)

    # measured/calculated from input traces
    input_traces = dict()
    input_traces["measured_trace"] = measured_trace
    input_traces["input_auto"] = input_auto
    input_traces["input_proof"] = input_proof

    # reconstruction traces
    recons_traces = dict()
    recons_traces["reconstructed"] = reconstructed
    recons_traces["reconstructed_proof"] = reconstructed_proof
    recons_traces["reconstruced_auto"] = reconstruced_auto

    if retrieval == "normal":
        plot_images_fields(axes=axes, traces_meas=input_traces["measured_trace"], traces_reconstructed=recons_traces["reconstructed"], xuv_f=xuv_f,
                            xuv_t=xuv_t, ir_f=ir_f, i=i, trace_yaxis=params.K, run_name=run_name)
        plt.pause(0.00001)

    elif retrieval == "proof":

        plot_images_fields(axes=axes, traces_meas=input_traces["input_proof"], traces_reconstructed=recons_traces["reconstructed_proof"], xuv_f=xuv_f,
                            xuv_t=xuv_t, ir_f=ir_f, i=i, trace_yaxis=params.K, run_name=run_name)
        plt.pause(0.00001)

    elif retrieval == "autocorrelation":

        plot_images_fields(axes=axes, traces_meas=input_traces["input_auto"], traces_reconstructed=recons_traces["reconstruced_auto"], xuv_f=xuv_f,
                            xuv_t=xuv_t, ir_f=ir_f, i=i, trace_yaxis=params.K, run_name=run_name)
        plt.pause(0.00001)


def create_plot_axes():

    fig = plt.figure(figsize=(8,7))
    fig.subplots_adjust(hspace=0.6, left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4)
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


if __name__ == "__main__":

    run_name = "known_trace_200ct"
    iterations = 2500

    #===================
    #==Retrieval Type===
    #===================
    retrieval = "normal"
    #retrieval = "autocorrelation"
    #retrieval = "proof"

    run_name = run_name + retrieval

    # copy the model to a new version to use for unsupervised learning
    modelname = "test1a"
    for file in glob.glob(r'./models/{}.ckpt.*'.format(modelname)):
        file_newname = file.replace(modelname, modelname+'_unsupervised')
        shutil.copy(file, file_newname)

    # get the measured trace
    # _, _, measured_trace = get_measured_trace()
    _, _, measured_trace = measured_trace.delay, measured_trace.energy, measured_trace.trace

    # get "measured" trace
    # measured_trace = get_fake_measured_trace(counts=200, plotting=True, run_name=run_name)
    # plt.show()


    # build neural net graph
    nn_nodes = network3.setup_neural_net()

    # create mse measurer
    writer = tf.summary.FileWriter("./tensorboard_graph_u/" + run_name)
    if retrieval == "normal":
        unsupervised_mse_tb = tf.summary.scalar("trace_mse", nn_nodes["unsupervised"]["unsupervised_learning_loss"])
    elif retrieval == "proof":
        unsupervised_mse_tb = tf.summary.scalar("trace_mse", nn_nodes["unsupervised"]["proof"]["proof_unsupervised_learning_loss"])
    elif retrieval == "autocorrelation":
        unsupervised_mse_tb = tf.summary.scalar("trace_mse", nn_nodes["unsupervised"]["autocorrelate"]["autocorrelate_unsupervised_learning_loss"])
    else:
        unsupervised_mse_tb = None

    # init data object
    get_data = network3.GetData(batch_size=10)


    axes = create_plot_axes()


    # =================================================
    # check the measured and training data proof traces
    # =================================================
    # with tf.Session() as sess:
    #
    #     # get a sample trace
    #     batch_x, batch_y = get_data.next_batch()
    #     trace_sample = batch_x[0].reshape(len(streak_params["p_values"]), len(streak_params["tau_values"]))
    #
    #     show_proof_calculation(trace=trace_sample, sess=sess, nn_nodes=nn_nodes)
    #     show_proof_calculation(trace=measured_trace, sess=sess, nn_nodes=nn_nodes)
    #
    #     plt.show()
    #
    #
    # exit(0)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './models/{}.ckpt'.format(modelname+'_unsupervised'))

        # get the initial output
        reconstruced = sess.run(nn_nodes["general"]["reconstructed_trace"],
                                feed_dict={nn_nodes["general"]["x_in"]: measured_trace.reshape(1, -1)})

        plt.ion()
        for i in range(iterations):

            if i % 10 == 0 or i == (iterations-1):

                print(i)
                # get MSE between traces
                summ = sess.run(unsupervised_mse_tb,
                                feed_dict={nn_nodes["general"]["x_in"]: measured_trace.reshape(1, -1)})
                writer.add_summary(summ, global_step=i + 1)
                writer.flush()

            if i % 500 == 0 or i == (iterations-1):
                # update plots
                update_plots(sess=sess, nn_nodes=nn_nodes, axes=axes, measured_trace=measured_trace, i=i+1,
                             retrieval=retrieval, run_name=run_name)

            # train neural network
            if retrieval == "normal":
                #========================
                #=========regular========
                #========================
                sess.run(nn_nodes["unsupervised"]["unsupervised_train"],
                         feed_dict={
                             nn_nodes["unsupervised"]["u_LR"]: 0.00001,
                             nn_nodes["unsupervised"]["x_in"]: measured_trace.reshape(1, -1),
                         })

            elif retrieval == "proof":
                # ========================
                # =========proof==========
                # ========================
                sess.run(nn_nodes["unsupervised"]["proof"]["proof_unsupervised_train"],
                         feed_dict={
                             nn_nodes["unsupervised"]["proof"]["u_LR"]: 0.00001,
                             nn_nodes["unsupervised"]["proof"]["x_in"]: measured_trace.reshape(1, -1),
                         })

            elif retrieval == "autocorrelation":
                # ========================
                # =========proof==========
                # ========================
                sess.run(nn_nodes["unsupervised"]["autocorrelate"]["autocorrelate_unsupervised_train"],
                         feed_dict={
                             nn_nodes["unsupervised"]["autocorrelate"]["u_LR"]: 0.00001,
                             nn_nodes["unsupervised"]["autocorrelate"]["x_in"]: measured_trace.reshape(1, -1),
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





