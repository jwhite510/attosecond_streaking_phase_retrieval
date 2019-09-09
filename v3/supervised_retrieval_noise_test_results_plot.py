import tensorflow as tf
from xuv_spectrum import spectrum
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tf_functions
import phase_parameters
import sys
modelname = sys.argv[1]
save_folder = "./9_5_19/"
# modelname = "DDD3normal_notanh2_long_512dense_leaky_activations_hp1_120ksamples_sample4_1_multires_stride"

def normal_text(ax, pos, text, ha=None):
    if ha is not None:
        ax.text(pos[0], pos[1], text, backgroundcolor="cyan", transform=ax.transAxes, ha=ha)
    else:
        ax.text(pos[0], pos[1], text, backgroundcolor="cyan", transform=ax.transAxes)



if __name__ == "__main__":

    with open(modelname+"_noise_test.p", "rb") as file:
        obj = pickle.load(file)

    xuv_coefs_in = tf.placeholder(tf.float32, shape=[None, phase_parameters.params.xuv_phase_coefs])
    generated_xuv = tf_functions.xuv_taylor_to_E(xuv_coefs_in)
    with tf.Session() as sess:

        j = 0
        fig1 = plt.figure(figsize=(15, 5))
        fig1.subplots_adjust(left=0.05, right=0.95, hspace=0.5)
        gs1 = fig1.add_gridspec(3, 5)

        fig2 = plt.figure(figsize=(15, 5))
        fig2.subplots_adjust(left=0.05, right=0.95, hspace=0.5)
        gs2 = fig2.add_gridspec(3, 5)

        fig3 = plt.figure(figsize=(15, 5))
        fig3.subplots_adjust(left=0.05, right=0.95, hspace=0.5)
        gs3 = fig3.add_gridspec(3, 5)

        fig4 = plt.figure(figsize=(15, 5))
        fig4.subplots_adjust(left=0.05, right=0.95, hspace=0.5)
        gs4 = fig4.add_gridspec(3, 5)

        for measured_trace, retrieved_coefs, count_num, xuv_input_coefs in zip(obj["measured_trace"], obj["retrieved_xuv_coefs"], obj["count_num"], obj["xuv_input_coefs"]):

            xuv_actual = sess.run(generated_xuv, feed_dict={xuv_coefs_in:xuv_input_coefs})
            retrieved = sess.run(generated_xuv, feed_dict={xuv_coefs_in:retrieved_coefs})

            if j < 5:
                # plot the trace
                ax = fig1.add_subplot(gs1[0, j])
                ax.pcolormesh(phase_parameters.params.delay_values_fs, phase_parameters.params.K, measured_trace, cmap="jet")
                ax.set_title("Count Number: {}".format(count_num))
                if j==0:
                    ax.set_ylabel("K")

                # plot actual the pulse in time
                ax = fig1.add_subplot(gs1[1, j])
                ax.plot(spectrum.tmat_as, np.abs(xuv_actual["t"][0])**2, color="black")
                ax.set_xlim(-800, 800)
                ax.set_xticks([])
                ax.set_yticks([])
                for k, coef in enumerate(xuv_input_coefs[0]):
                    normal_text(ax, (0.0 + k*0.2, 1.1), "%.2f" % xuv_input_coefs[0][k])
                if j == 0:
                    ax.set_ylabel("Actual\nIntensity")

                # plot actual the pulse in time
                ax = fig1.add_subplot(gs1[2, j])
                ax.plot(spectrum.tmat_as, np.abs(retrieved["t"][0])**2, color="black")
                ax.set_xlim(-800, 800)
                ax.set_yticks([])
                ax.set_xlabel("time [as]")
                for k, coef in enumerate(retrieved_coefs[0]):
                    normal_text(ax, (0.0 + k*0.2, 1.1), "%.2f" % retrieved_coefs[0][k])
                if j == 0:
                    ax.set_ylabel("Retrieved\nIntensity")
            elif j < 10:
                # plot the trace
                ax = fig2.add_subplot(gs2[0, (j-5)])
                ax.pcolormesh(phase_parameters.params.delay_values_fs, phase_parameters.params.K, measured_trace, cmap="jet")
                ax.set_title("Count Number: {}".format(count_num))
                if (j-5)==0:
                    ax.set_ylabel("K")

                # plot actual the pulse in time
                ax = fig2.add_subplot(gs2[1, (j-5)])
                ax.plot(spectrum.tmat_as, np.abs(xuv_actual["t"][0])**2, color="black")
                ax.set_xlim(-800, 800)
                ax.set_xticks([])
                ax.set_yticks([])
                for k, coef in enumerate(xuv_input_coefs[0]):
                    normal_text(ax, (0.0 + k*0.2, 1.1), "%.2f" % xuv_input_coefs[0][k])
                if (j-5) == 0:
                    ax.set_ylabel("Actual\nIntensity")

                # plot actual the pulse in time
                ax = fig2.add_subplot(gs2[2, (j-5)])
                ax.plot(spectrum.tmat_as, np.abs(retrieved["t"][0])**2, color="black")
                ax.set_xlim(-800, 800)
                ax.set_yticks([])
                ax.set_xlabel("time [as]")
                for k, coef in enumerate(retrieved_coefs[0]):
                    normal_text(ax, (0.0 + k*0.2, 1.1), "%.2f" % retrieved_coefs[0][k])
                if (j-5) == 0:
                    ax.set_ylabel("Retrieved\nIntensity")
            elif j < 15:
                # plot the trace
                ax = fig3.add_subplot(gs3[0, (j-10)])
                ax.pcolormesh(phase_parameters.params.delay_values_fs, phase_parameters.params.K, measured_trace, cmap="jet")
                ax.set_title("Count Number: {}".format(count_num))
                if (j-10)==0:
                    ax.set_ylabel("K")

                # plot actual the pulse in time
                ax = fig3.add_subplot(gs3[1, (j-10)])
                ax.plot(spectrum.tmat_as, np.abs(xuv_actual["t"][0])**2, color="black")
                ax.set_xlim(-800, 800)
                ax.set_xticks([])
                ax.set_yticks([])
                for k, coef in enumerate(xuv_input_coefs[0]):
                    normal_text(ax, (0.0 + k*0.2, 1.1), "%.2f" % xuv_input_coefs[0][k])
                if (j-10) == 0:
                    ax.set_ylabel("Actual\nIntensity")

                # plot actual the pulse in time
                ax = fig3.add_subplot(gs3[2, (j-10)])
                ax.plot(spectrum.tmat_as, np.abs(retrieved["t"][0])**2, color="black")
                ax.set_xlim(-800, 800)
                ax.set_yticks([])
                ax.set_xlabel("time [as]")
                for k, coef in enumerate(retrieved_coefs[0]):
                    normal_text(ax, (0.0 + k*0.2, 1.1), "%.2f" % retrieved_coefs[0][k])
                if (j-10) == 0:
                    ax.set_ylabel("Retrieved\nIntensity")
            elif j < 20:
                # plot the trace
                ax = fig4.add_subplot(gs4[0, (j-15)])
                ax.pcolormesh(phase_parameters.params.delay_values_fs, phase_parameters.params.K, measured_trace, cmap="jet")
                ax.set_title("Count Number: {}".format(count_num))
                if (j-15)==0:
                    ax.set_ylabel("K")

                # plot actual the pulse in time
                ax = fig4.add_subplot(gs4[1, (j-15)])
                ax.plot(spectrum.tmat_as, np.abs(xuv_actual["t"][0])**2, color="black")
                ax.set_xlim(-800, 800)
                ax.set_xticks([])
                ax.set_yticks([])
                for k, coef in enumerate(xuv_input_coefs[0]):
                    normal_text(ax, (0.0 + k*0.2, 1.1), "%.2f" % xuv_input_coefs[0][k])
                if (j-15) == 0:
                    ax.set_ylabel("Actual\nIntensity")

                # plot actual the pulse in time
                ax = fig4.add_subplot(gs4[2, (j-15)])
                ax.plot(spectrum.tmat_as, np.abs(retrieved["t"][0])**2, color="black")
                ax.set_xlim(-800, 800)
                ax.set_yticks([])
                ax.set_xlabel("time [as]")
                for k, coef in enumerate(retrieved_coefs[0]):
                    normal_text(ax, (0.0 + k*0.2, 1.1), "%.2f" % retrieved_coefs[0][k])
                if (j-15) == 0:
                    ax.set_ylabel("Retrieved\nIntensity")

            # # measured trace
            # measured_trace
            # # retrieved E
            # retrieved
            # # actual E
            # xuv_actual
            # # count number
            # count_num

            # just add plotting

            j+=1

        fig1.savefig(save_folder+modelname+"_noise_test1f.png")
        fig2.savefig(save_folder+modelname+"_noise_test2f.png")
        fig3.savefig(save_folder+modelname+"_noise_test3f.png")
        fig4.savefig(save_folder+modelname+"_noise_test4f.png")
        # plt.show()

        # plot the measured trace retrieval
        # this list should always be length of 1...
        with open(modelname+"_noise_test_measured.p", "rb") as file:
            obj_meas = pickle.load(file)

        measured_trace = obj_meas["measured_trace"]
        retrieved_coefs = obj_meas["retrieved_xuv_coefs"]
        retrieved = sess.run(generated_xuv, feed_dict={xuv_coefs_in:retrieved_coefs})

        meas_fig = plt.figure(figsize=(15, 5))
        meas_fig.subplots_adjust(left=0.05, right=0.95, hspace=0.5)
        gs_meas_fig = meas_fig.add_gridspec(1, 2)

        ax = meas_fig.add_subplot(gs_meas_fig[0, 0])
        ax.pcolormesh(phase_parameters.params.delay_values_fs, phase_parameters.params.K, measured_trace, cmap="jet")

        # plot retrieved signal in time
        ax = meas_fig.add_subplot(gs_meas_fig[0, 1])
        ax.plot(spectrum.tmat_as, np.abs(retrieved["t"][0])**2, color="black")
        ax.set_xlim(-800, 800)
        ax.set_yticks([])
        ax.set_xlabel("time [as]")
        for k, coef in enumerate(retrieved_coefs[0]):
            normal_text(ax, (0.0 + k*0.2, 1.1), "%.2f" % retrieved_coefs[0][k])
        ax.set_ylabel("Retrieved\nIntensity")
        meas_fig.savefig(save_folder+modelname+"_meas.png")

