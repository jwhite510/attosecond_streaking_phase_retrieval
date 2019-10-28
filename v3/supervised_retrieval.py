import pickle
import tensorflow as tf
from ir_spectrum import ir_spectrum
import os
import scipy.constants as sc
from xuv_spectrum import spectrum
import matplotlib.pyplot as plt
import numpy as np
import tf_functions
import sys
# modelname = "DDD3normal_notanh2_long_512dense_leaky_activations_hp1_120ksamples_sample4_1_multires_stride"
test_run = "noise_test_1"
import importlib
from phase_parameters import params
import measured_trace.get_trace as get_measured_trace

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

def normal_text(ax, pos, text, ha=None):
    if ha is not None:
        ax.text(pos[0], pos[1], text, backgroundcolor="white", transform=ax.transAxes, ha=ha)
    else:
        ax.text(pos[0], pos[1], text, backgroundcolor="white", transform=ax.transAxes)

def plot_images_fields(axes, traces_meas, traces_reconstructed, xuv_f, xuv_f_phase,  xuv_f_full, xuv_t, ir_f, i,
                       run_name, true_fields=False, cost_function=None, method=None, save_data_objs=False):

    if save_data_objs:
        file_objs = dict()
        file_objs["axes"] = axes
        file_objs["traces_meas"] = traces_meas
        file_objs["traces_reconstructed"] = traces_reconstructed
        file_objs["xuv_f"] = xuv_f
        file_objs["xuv_f_phase"] = xuv_f_phase
        file_objs["xuv_f_full"] = xuv_f_full
        file_objs["xuv_t"] = xuv_t
        file_objs["ir_f"] = ir_f
        file_objs["i"] = i
        file_objs["run_name"] = run_name 
        file_objs["true_fields"] = true_fields 
        file_objs["cost_function"] = cost_function 
        file_objs["method"] = method 

    # ...........................
    # ........CLEAR AXES.........
    # ...........................
    # input trace
    axes["input_normal_trace"].cla()
    axes["input_proof_trace"].cla()
    axes["input_auto_trace"].cla()
    # generated trace
    axes["generated_normal_trace"].cla()
    axes["generated_proof_trace"].cla()
    axes["generated_auto_trace"].cla()
    # xuv predicted
    axes["predicted_xuv_t"].cla()
    axes["predicted_xuv"].cla()
    axes["predicted_xuv_phase"].cla()
    # predicted ir
    axes["predicted_ir"].cla()
    axes["predicted_ir_phase"].cla()
    # ...........................
    # .....CALCULATE RMSE........
    # ...........................
    # calculate the rmse for each trace
    rmses = dict()
    for trace_type in ["trace", "autocorrelation", "proof"]:
        rmse = np.sqrt((1 / len(traces_meas[trace_type].reshape(-1))) * np.sum(
            (traces_meas[trace_type].reshape(-1) - traces_reconstructed[trace_type].reshape(-1)) ** 2))
        rmses[trace_type] = rmse

    # .......................................
    # .......................................
    # .......................................
    # ...............PLOTTING................
    # .......................................
    # .......................................
    # .......................................


    # just for testing
    # cost_function = "autocorrelation"
    # true_fields = False

    # ..........................................
    # ...............input traces...............
    # ..........................................
    axes["input_normal_trace"].pcolormesh(params.delay_values_fs, params.K, traces_meas["trace"], cmap='jet')
    axes["input_normal_trace"].set_xlabel(r"$\tau$ Delay [fs]")
    axes["input_normal_trace"].set_ylabel("Energy [eV]")
    if true_fields:
        normal_text(axes["input_normal_trace"], (0.0, 1.0), "noisy trace")
    else:
        normal_text(axes["input_normal_trace"], (0.0, 1.0), "input trace")
        if cost_function == "trace":
            red_text(axes["input_normal_trace"], (1.0, 1.0), "C")

    axes["input_proof_trace"].pcolormesh(params.delay_values_fs, params.K, traces_meas["proof"], cmap='jet')
    axes["input_proof_trace"].set_xlabel(r"$\tau$ Delay [fs]")
    axes["input_proof_trace"].set_ylabel("Energy [eV]")
    if true_fields:
        normal_text(axes["input_proof_trace"], (0.0, 1.0), "noisy proof trace")
        normal_text(axes["input_proof_trace"], (0.5, 1.2), "Actual Fields", ha="center")
    else:
        normal_text(axes["input_proof_trace"], (0.0, 1.0), "input proof trace")
        if method is not None:
            normal_text(axes["input_proof_trace"], (0.5, 1.2), method, ha="center")
        if cost_function == "proof":
            red_text(axes["input_proof_trace"], (1.0, 1.0), "C")

    if i is not None:
        if method == "Genetic Algorithm":
            normal_text(axes["input_proof_trace"], (1.3, 1.2), "Generation: " + str(i), ha="center")
        elif method == "Unsupervised Learning":
            normal_text(axes["input_proof_trace"], (1.3, 1.2), "Iteration: " + str(i), ha="center")
        elif method == "Training":
            normal_text(axes["input_proof_trace"], (1.3, 1.2), "Epoch: " + str(i), ha="center")
        else:
            raise ValueError("method should be unsupervised learning or genetic algorithm")



    axes["input_auto_trace"].pcolormesh(params.delay_values_fs, params.delay_values_fs, traces_meas["autocorrelation"], cmap='jet')
    axes["input_auto_trace"].set_xlabel(r"$\tau$ Delay [fs]")
    axes["input_auto_trace"].set_ylabel(r"$\tau$ Delay [fs]")
    if true_fields:
        normal_text(axes["input_auto_trace"], (0.0, 1.0), "noisy autocorrelation")
    else:
        normal_text(axes["input_auto_trace"], (0.0, 1.0), "input autocorrelation")
        if cost_function == "autocorrelation":
            red_text(axes["input_auto_trace"], (1.0, 1.0), "C")

    # ..........................................
    # ...............generated..................
    # ..........................................
    axes["generated_normal_trace"].pcolormesh(params.delay_values_fs, params.K, traces_reconstructed["trace"], cmap='jet')
    axes["generated_normal_trace"].set_xlabel(r"$\tau$ Delay [fs]")
    axes["generated_normal_trace"].set_ylabel("Energy [eV]")
    normal_text(axes["generated_normal_trace"], (0.05, 0.05), "RMSE: "+"%.4f" % rmses["trace"])
    if true_fields:
        normal_text(axes["generated_normal_trace"], (0.0, 1.0), "actual trace")
    else:
        normal_text(axes["generated_normal_trace"], (0.0, 1.0), "generated trace")
        if cost_function == "trace":
            red_text(axes["generated_normal_trace"], (1.0, 1.0), "C")

    axes["generated_proof_trace"].pcolormesh(params.delay_values_fs, params.K, traces_reconstructed["proof"], cmap='jet')
    axes["generated_proof_trace"].set_xlabel(r"$\tau$ Delay [fs]")
    axes["generated_proof_trace"].set_ylabel("Energy [eV]")
    normal_text(axes["generated_proof_trace"], (0.05, 0.05), "RMSE: "+"%.4f" % rmses["proof"])
    if true_fields:
        normal_text(axes["generated_proof_trace"], (0.0, 1.0), "proof trace")
    else:
        normal_text(axes["generated_proof_trace"], (0.0, 1.0), "generated proof trace")
        if cost_function == "proof":
            red_text(axes["generated_proof_trace"], (1.0, 1.0), "C")

    axes["generated_auto_trace"].pcolormesh(params.delay_values_fs, params.delay_values_fs, traces_reconstructed["autocorrelation"], cmap='jet')
    axes["generated_auto_trace"].set_xlabel(r"$\tau$ Delay [fs]")
    axes["generated_auto_trace"].set_ylabel(r"$\tau$ Delay [fs]")
    normal_text(axes["generated_auto_trace"], (0.05, 0.05), "RMSE: "+"%.4f" % rmses["autocorrelation"])
    if true_fields:
        normal_text(axes["generated_auto_trace"], (0.0, 1.0), "autocorrelation")
    else:
        normal_text(axes["generated_auto_trace"], (0.0, 1.0), "generated autocorrelation")
        if cost_function == "autocorrelation":
            red_text(axes["generated_auto_trace"], (1.0, 1.0), "C")

    # xuv f
    fmat_hz = spectrum.fmat_cropped/sc.physical_constants['atomic unit of time'][0]*1e-17
    axes["predicted_xuv"].plot(fmat_hz, np.abs(xuv_f) ** 2, color="black")
    axes["predicted_xuv"].set_yticks([])
    axes["predicted_xuv"].set_xlabel("Frequency [$10^{17}$Hz]")
    # plotting photon spectrum
    axes["predicted_xuv"].plot(fmat_hz, np.abs(spectrum.Ef_photon[spectrum.indexmin:spectrum.indexmax]) ** 2, color="blue")


    if true_fields:
        axes["predicted_xuv_phase"].text(0.0, 1.1, "actual XUV spectrum", backgroundcolor="white",
                                         transform=axes["predicted_xuv_phase"].transAxes)
    else:
        axes["predicted_xuv_phase"].text(0.0, 1.1, "predicted XUV spectrum", backgroundcolor="white",
                                         transform=axes["predicted_xuv_phase"].transAxes)

    axes["predicted_xuv_phase"].tick_params(axis='y', colors='green')
    axes["predicted_xuv_phase"].plot(fmat_hz, xuv_f_phase, color="green")


    # xuv predicted
    # xuv t
    tmat_as = spectrum.tmat * sc.physical_constants['atomic unit of time'][0] * 1e18

    # from the electron spectrum
    # I_t = np.abs(xuv_t) ** 2

    # from photon spectrum
    angle = np.angle(xuv_f_full)
    Ef_photon_phase = spectrum.Ef_photon * np.exp(1j * angle)
    Et_photon_phase = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ef_photon_phase)))
    I_t = np.abs(Et_photon_phase) ** 2

    axes["predicted_xuv_t"].plot(tmat_as, I_t, color="black")
    #calculate FWHM
    fwhm, t1, t2, half_max = calc_fwhm(tmat=tmat_as, I_t=I_t)
    axes["predicted_xuv_t"].text(1.0, 0.9, "FWHM:\n %.2f [as]" % fwhm, color="red",
                            backgroundcolor="white", ha="center",
                            transform=axes["predicted_xuv_t"].transAxes)
    #plot FWHM
    axes["predicted_xuv_t"].plot([t1, t2], [half_max, half_max], color="red", linewidth=2.0)
    axes["predicted_xuv_t"].set_yticks([])
    axes["predicted_xuv_t"].set_xlabel("time [as]")
    # axes["predicted_xuv_t"].set_xlim(-200, 300)

    if true_fields:
        axes["predicted_xuv_t"].text(0.0, 1.1, "actual XUV $I(t)$", backgroundcolor="white",
                                     transform=axes["predicted_xuv_t"].transAxes)
    else:
        axes["predicted_xuv_t"].text(0.0, 1.1, "predicted XUV $I(t)$", backgroundcolor="white",
                                     transform=axes["predicted_xuv_t"].transAxes)

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
        dir = "./retrieval/" + run_name + "/"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        axes["fig"].savefig(dir + str(i) + ".png")
        with open("./retrieval/" + run_name + "/u_fields.p", "wb") as file:
            predicted_fields = {}
            predicted_fields["ir_f"] = ir_f
            predicted_fields["xuv_f"] = xuv_f
            predicted_fields["xuv_t"] = xuv_t

            save_files = {}
            save_files["predicted_fields"] = predicted_fields
            save_files["traces_meas"] = traces_meas
            save_files["traces_reconstructed"] = traces_reconstructed
            save_files["i"] = i
            pickle.dump(save_files, file)

        # save the objects used to make the plot
        if save_data_objs:
            with open("./retrieval/" + run_name + "/plot_objs_epoch"+str(i)+".p", "wb") as file:
                pickle.dump(file_objs, file)

def apply_noise(trace, counts):
    discrete_trace = np.round(trace * counts)
    noise = np.random.poisson(lam=discrete_trace) - discrete_trace
    noisy_trace = discrete_trace + noise
    noisy_trace_normalized = noisy_trace / np.max(noisy_trace)
    return noisy_trace_normalized

def create_plot_axes():
    fig = plt.figure(figsize=(8,7))
    fig.subplots_adjust(hspace=0.6, left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4)
    gs = fig.add_gridspec(3, 3)

    axes_dict = dict()
    axes_dict["input_normal_trace"] = fig.add_subplot(gs[0,0])
    axes_dict["input_proof_trace"] = fig.add_subplot(gs[0,1])
    axes_dict["input_auto_trace"] = fig.add_subplot(gs[0,2])

    axes_dict["predicted_xuv_t"] = fig.add_subplot(gs[1, 2])

    axes_dict["predicted_xuv"] = fig.add_subplot(gs[1,1])
    axes_dict["predicted_xuv_phase"] = axes_dict["predicted_xuv"].twinx()

    axes_dict["predicted_ir"] = fig.add_subplot(gs[1,0])
    axes_dict["predicted_ir_phase"] = axes_dict["predicted_ir"].twinx()

    axes_dict["generated_normal_trace"] = fig.add_subplot(gs[2,0])
    axes_dict["generated_proof_trace"] = fig.add_subplot(gs[2,1])
    axes_dict["generated_auto_trace"] = fig.add_subplot(gs[2,2])
    axes_dict["fig"] = fig

    return axes_dict

def get_fake_measured_trace(counts, plotting, run_name=None):
    # initialize XUV generator
    xuv_coefs_in = tf.placeholder(tf.float32, shape=[None, params.xuv_phase_coefs])
    xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_in)

    # initialize IR generator
    ir_values_in = tf.placeholder(tf.float32, shape=[None, 4])
    ir_E_prop = tf_functions.ir_from_params(ir_values_in)["E_prop"]

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

    xuv_input = np.array([[0.0, -0.1, 0.2, 0.2, 0.0]])
    ir_input = np.array([[0.0, 0.0, 0.0, 0.0]])

    with tf.Session() as sess:
        feed_dict = {tf_graphs["xuv_coefs_in"]: xuv_input, tf_graphs["ir_values_in"]: ir_input}
        trace = sess.run(tf_graphs["image"], feed_dict=feed_dict)
        trace_proof = sess.run(tf_graphs["proof_trace"]["proof"], feed_dict=feed_dict)
        trace_autocorrelation = sess.run(tf_graphs["autocorelation"], feed_dict=feed_dict)
        xuv_t = sess.run(xuv_E_prop["t"], feed_dict=feed_dict)[0]
        xuv_f = sess.run(xuv_E_prop["f_cropped"], feed_dict=feed_dict)[0]
        phase_curve = sess.run(xuv_E_prop["phasecurve_cropped"], feed_dict=feed_dict)[0]
        xuv_f_full = sess.run(xuv_E_prop["f"], feed_dict=feed_dict)[0]
        ir_f = sess.run(ir_E_prop["f_cropped"], feed_dict=feed_dict)[0]
        # construct proof and autocorrelate from non-noise trace

    # generated from noise free trace
    traces = {}
    traces["trace"] = trace
    traces["autocorrelation"] = trace_autocorrelation
    traces["proof"] = trace_proof

    if counts == 0:
        # no noise
        noisy_trace = trace
    else:
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

    axes = None
    if plotting:
        axes = create_plot_axes()
        plot_images_fields(axes=axes, traces_meas=noise_traces, traces_reconstructed=traces,
                           xuv_f=xuv_f, xuv_f_phase=phase_curve, xuv_f_full=xuv_f_full, xuv_t=xuv_t, ir_f=ir_f, i=None,
                           run_name=None, true_fields=True, cost_function="trace")

        # save files
        dir = "./retrieval/" + run_name + "/"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        plt.savefig(dir+"actual_fields" + str(counts) + ".png")

    return noisy_trace, phase_curve, axes, xuv_input


class SupervisedRetrieval:
    def __init__(self, model):
        """
        a class for taking only the initial output of the neural network, no additional changing the weights
        """

        self.modelname = model
        self.network3 = importlib.import_module("models.network3_"+self.modelname)
        # build neural net graph
        self.nn_nodes = self.network3.setup_neural_net()

        # restore session
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, './models/{}.ckpt'.format(self.modelname))

    def retrieve(self, trace):

        self.feed_dict = {self.nn_nodes["general"]["x_in"]: trace.reshape(1, -1)}
        # return self.sess.run(self.nn_nodes["general"]["phase_net_output"]["xuv_E_prop"], feed_dict=self.feed_dict)
        trace_recons = self.sess.run(self.nn_nodes["general"]["reconstructed_trace"], feed_dict=self.feed_dict)
        xuv_retrieved = self.sess.run(self.nn_nodes["general"]["xuv_coefs_pred"], feed_dict=self.feed_dict)
        ir_params_pred = self.sess.run(self.nn_nodes["general"]["phase_net_output"]["predicted_coefficients_params"], feed_dict=self.feed_dict)[:, params.xuv_phase_coefs:]
        predicted_coefficients_params = self.sess.run(self.nn_nodes["general"]["phase_net_output"]["predicted_coefficients_params"], feed_dict=self.feed_dict)


        retrieve_output = {}
        retrieve_output["trace_recons"] = trace_recons
        retrieve_output["xuv_retrieved"] = xuv_retrieved
        retrieve_output["ir_params_pred"] = ir_params_pred
        retrieve_output["predicted_coefficients_params"] = predicted_coefficients_params

        return retrieve_output

    def __del__(self):
        self.sess.close()
        tf.reset_default_graph()


if __name__ == "__main__":

    modelname = sys.argv[1]
    # test retrieval after supervised learning on different noise levels
    # noise_test_initial_only("supervised_learning_noise_test")

    snr_min = np.sqrt(20)  # minimum count level
    snr_max = np.sqrt(5000)  # maximum count level
    snr_levels = np.linspace(snr_min, snr_max, 40)
    # counts_list = [int(count) for count in snr_levels**2]

    supervised_retrieval = SupervisedRetrieval(modelname)
    retrieval_data = {}
    retrieval_data["measured_trace"] = []
    retrieval_data["retrieved_xuv_coefs"] = []
    retrieval_data["count_num"] = []
    retrieval_data["xuv_input_coefs"] = []



    # for counts in counts_list:
    # make the same as in the generated data set
    counts_min, counts_max = 25, 200

    # get traces from validations set
    get_data = supervised_retrieval.network3.GetData(batch_size=10)
    batch_x_test, batch_y_test = get_data.evaluate_on_test_data()

    # just the xuv coefficients
    xuv_coefs_actual = batch_y_test[:,0:5]

    # for counts in np.linspace(counts_min, counts_max, 5):
    index_min = 25
    index_max = 30
    # this is from generate_data3.py line 224
    counts_min, counts_max = 25, 200
    counts_values = np.linspace(counts_min, counts_max, 5)
    for trace, xuv_coefs, counts in zip(batch_x_test[index_min:index_max], xuv_coefs_actual[index_min:index_max], counts_values):

        K_values = params.K
        tau_values = params.delay_values
        measured_trace = trace.reshape(len(K_values), len(tau_values))
        xuv_input_coefs = xuv_coefs.reshape(1, -1)
        # run_name = test_run + str(counts)


        retrieved_xuv_coefs = supervised_retrieval.retrieve(measured_trace)["xuv_retrieved"]
        # print(counts)
        print("retrieved xuv")
        retrieval_data["measured_trace"].append(measured_trace)
        retrieval_data["retrieved_xuv_coefs"].append(retrieved_xuv_coefs)
        retrieval_data["count_num"].append(counts)
        retrieval_data["xuv_input_coefs"].append(xuv_input_coefs)

    print("saving pickle")
    with open(modelname+"_noise_test.p", "wb") as file:
        pickle.dump(retrieval_data, file)

    # retrieval with measured trace
    K_values = params.K
    tau_values = params.delay_values
    measured_trace = get_measured_trace.trace
    # this measured trace: 301, 98
    retrieve_output = supervised_retrieval.retrieve(measured_trace)
    retrieved_xuv_coefs = retrieve_output["xuv_retrieved"]
    reconstructed_trace = retrieve_output["trace_recons"]

    retrieval_data_measured_trace = {}
    retrieval_data_measured_trace["measured_trace"] = measured_trace
    retrieval_data_measured_trace["retrieved_xuv_coefs"] = retrieved_xuv_coefs
    retrieval_data_measured_trace["reconstructed_trace"] = reconstructed_trace

    print("saving pickle of measured trace")
    with open(modelname+"_noise_test_measured.p", "wb") as file:
        pickle.dump(retrieval_data_measured_trace, file)
