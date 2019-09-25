import tensorflow as tf
import xuv_spectrum.spectrum
import ir_spectrum.ir_spectrum
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial
import scipy.constants as sc
import math
import phase_parameters.params
# import generate_data3
import pickle
# import unsupervised_retrieval
import imageio


class TestGraphs:

    def __init__(self):
        self.sess = tf.Session()

        # initialize graphs
        # xuv creation
        self.xuv_coefs_in = tf.placeholder(tf.float32, shape=[None, phase_parameters.params.xuv_phase_coefs])
        self.xuv_E_prop = xuv_taylor_to_E(self.xuv_coefs_in)

        self.ir_values_in = tf.placeholder(tf.float32, shape=[None, 4])
        self.ir_E_prop = ir_from_params(self.ir_values_in)["E_prop"]
        # image1, _ = streaking_trace_old(xuv_cropped_f_in=self.xuv_E_prop["f_cropped"][0], ir_cropped_f_in=self.ir_E_prop["f_cropped"][0])
        # construct streaking image
        self.image2 = streaking_traceA(xuv_cropped_f_in=self.xuv_E_prop["f_cropped"][0], ir_cropped_f_in=self.ir_E_prop["f_cropped"][0])
        self.image2_2 = streaking_trace(xuv_cropped_f_in=self.xuv_E_prop["f_cropped"][0], ir_cropped_f_in=self.ir_E_prop["f_cropped"][0])

        # construct proof trace
        self.proof2 = proof_trace(self.image2_2)

        self.autocorrelateion2 = autocorrelate(self.image2)


    def test_coef_scale(self):
        pass


    def plot_xuv_trace(self, feed_dict_in):

        feed_dict = {
            self.xuv_coefs_in: feed_dict_in["xuv_coefs_in"],
            self.ir_values_in: feed_dict_in["ir_values_in"]
        }


        f_cropped_fmat = xuv_spectrum.spectrum.fmat_cropped
        tmat_xuv = xuv_spectrum.spectrum.tmat_as
        xuv_out = self.sess.run(self.xuv_E_prop, feed_dict=feed_dict)
        trace = self.sess.run(self.image2_2, feed_dict=feed_dict)
        proof = self.sess.run(self.proof2["proof"], feed_dict=feed_dict)
        fig = plt.figure(figsize=(10,10))
        gs = fig.add_gridspec(2,2)

        # xuv (t)
        ax_xuv = fig.add_subplot(gs[0,0])
        ax_xuv.plot(tmat_xuv, np.real(xuv_out["t"][0]), color="blue")
        ax_xuv.set_xlabel("attoseconds")
        ax_xuv.set_title("Real $E(t)$")
        ax_xuv.set_yticks([])

        # xuv I(t)
        ax_ir = fig.add_subplot(gs[0,1])
        ax_ir.plot(tmat_xuv, np.abs(xuv_out["t"][0])**2, color="black")
        ax_ir.set_xlabel("attoseconds")
        ax_ir.set_title("$I(t)$")
        ax_ir.set_yticks([])

        # show xuv coef values
        for ax in [ax_ir, ax_xuv]:
            vpos = 0.9
            for value, type in zip(feed_dict_in["xuv_coefs_in"].reshape(-1), ["1st", "2nd", "3rd", "4th", "5th"]):
                normal_text(ax, (0.75, vpos), type, ha="center")
                normal_text(ax, (0.9, vpos), "%.2f" % value, ha="center")
                vpos -= 0.05

        # spectrum
        ax_xuvf = fig.add_subplot(gs[1,0])
        ax_xuvf.plot(xuv_spectrum.spectrum.fmat_hz_cropped, np.abs(xuv_out["f_cropped"][0])**2, color="black")
        ax_xuvf.set_yticks([])
        ax_xuvf.set_xlabel("Hz")
        ax_xuv_phase = ax_xuvf.twinx()
        ax_xuv_phase.plot(xuv_spectrum.spectrum.fmat_hz_cropped, np.unwrap(np.angle(xuv_out["f_cropped"][0])), color="green")
        ax_xuv_phase.tick_params(axis="y", colors="green")

        # trace
        ax = fig.add_subplot(gs[1,1])
        ax.pcolormesh(phase_parameters.params.delay_values_fs,phase_parameters.params.K, trace, cmap="jet")
        ax.yaxis.tick_right()
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel("eV")
        fig.savefig("./5.png")
        exit(0)

    def __del__(self):
        self.sess.close()


def normal_text(ax, pos, text, ha=None):

    if ha is not None:
        ax.text(pos[0], pos[1], text, backgroundcolor="white", transform=ax.transAxes, ha=ha)
    else:
        ax.text(pos[0], pos[1], text, backgroundcolor="white", transform=ax.transAxes)


def set_point_to(vector, index, value):
    diff = vector[index] - value
    vector = vector - diff
    return vector



def animate_trace(sess, xuv_coefs_in, ir_values_in, xuv_E_prop, image2_2):
    # make graph
    fig = plt.figure(figsize=(17, 5))
    fig.subplots_adjust(wspace=0.4, left=0.05, right=0.95)
    gs = fig.add_gridspec(2, 4)
    plt.ion()

    # create axes
    axes = {}
    axes["xuv_Et"] = fig.add_subplot(gs[0, 0])
    axes["xuv_It"] = fig.add_subplot(gs[1, 0])
    axes["xuv_f"] = fig.add_subplot(gs[0, 1])
    axes["xuv_f_phase"] = axes["xuv_f"].twinx()
    axes["trace"] = fig.add_subplot(gs[0:2, 2])
    axes["trace_meas"] = fig.add_subplot(gs[0:2, 3])
    # plot the measured trace
    delay, energy, measured_trace = unsupervised_retrieval.get_measured_trace()
    axes["trace_meas"].pcolormesh(delay * 1e15, energy, measured_trace, cmap="jet")
    axes["trace_meas"].set_title("measured trace")

    # calculate feed dicts
    feed_dicts = []
    step = 0.2
    gdd_vals = -1 * np.arange(0, 5.0 + step, step)
    for val in gdd_vals:
        feed_dict_i = {
            xuv_coefs_in: np.array([[0.0, val, 0.0, 0.0, 0.0]]),
            ir_values_in: np.array([[1.0, 0.0, 0.0, 0.0]])
        }
        feed_dicts.append(feed_dict_i)

    gif_images = []

    for feed_dict in feed_dicts:
        # generate output
        xuv_out = sess.run(xuv_E_prop, feed_dict=feed_dict)
        out_2 = sess.run(image2_2, feed_dict=feed_dict)
        # plot output
        axes["xuv_Et"].cla()
        axes["xuv_Et"].plot(np.real(xuv_out["t"][0]), color="blue")
        axes["xuv_Et"].set_title("$E(t)$")
        axes["xuv_Et"].set_xticks([])
        axes["xuv_Et"].set_ylim(-0.05, 0.05)

        axes["xuv_It"].cla()
        xuv_time_fs = xuv_spectrum.spectrum.tmat * sc.physical_constants['atomic unit of time'][0] * 1e18
        I_t = np.abs(xuv_out["t"][0]) ** 2
        axes["xuv_It"].plot(xuv_time_fs,
                            I_t, color="black")

        # calc fwhm
        halfmaxI = np.max(I_t) / 2
        I2_I = np.abs(I_t - halfmaxI)
        sorted = np.argsort(I2_I)
        index1 = sorted[0]
        index2 = find_second_minima(sorted, index1)
        fwhm = np.abs(xuv_time_fs[index1] - xuv_time_fs[index2])
        axes["xuv_It"].plot([xuv_time_fs[index1], xuv_time_fs[index2]], [halfmaxI, halfmaxI], color="red", linewidth=2)
        axes["xuv_It"].text(0.7, 0.8, "FWHM [as]: " + str(round(fwhm, 2)), transform=axes["xuv_It"].transAxes,
                            color="red",
                            backgroundcolor="white")
        axes["xuv_It"].set_xlabel("time [as]")
        axes["xuv_It"].set_title("$I(t)$")
        axes["xuv_It"].set_ylim(0, 1.1 * np.max(I_t))

        axes["xuv_f"].cla()
        xuv_f_hz = xuv_spectrum.spectrum.fmat_cropped / sc.physical_constants['atomic unit of time'][0]
        xuv_f_hz = xuv_f_hz * 1e-17
        axes["xuv_f"].plot(xuv_f_hz, np.abs(xuv_out["f_cropped"][0]) ** 2, color="black")
        axes["xuv_f"].set_title("XUV spectral phase")
        axes["xuv_f"].set_xlabel("frequency [$10^{17}$Hz]")
        axes["xuv_f_phase"].cla()
        axes["xuv_f_phase"].plot(xuv_f_hz, xuv_out["phasecurve_cropped"][0], color="green")
        axes["xuv_f_phase"].tick_params(axis='y', colors='green')
        axes["xuv_f_phase"].set_ylim(-20, 20)

        axes["trace"].cla()
        axes["trace"].pcolormesh(
            phase_parameters.params.delay_values * sc.physical_constants['atomic unit of time'][0] * 1e15,
            phase_parameters.params.K,
            out_2, cmap="jet")
        axes["trace"].set_xlabel("Delay [fs]")
        axes["trace"].set_ylabel("Energy [eV]")

        textdraw = "_XUV Phase_"
        for type, phasecoef in zip(["1", "2", "3", "4", "5"], feed_dict[xuv_coefs_in][0]):
            textdraw += "\n" + "$\phi$" + type + " : " + '%.2f' % phasecoef
        axes["xuv_f_phase"].text(0.5, -1.0, textdraw, ha="center", transform=axes["xuv_f_phase"].transAxes)

        plt.pause(0.001)
        fig.canvas.draw()
        image_draw = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image_draw = image_draw.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif_images.append(image_draw)

    print("making gif")
    imageio.mimsave('./A2_2.gif', gif_images, fps=10)


def compare_A_A2_animate(sess, xuv_coefs_in, ir_values_in, xuv_E_prop, image2, image2_2):
    # ===============================================
    # =======testing trace difference A/A^2==========
    # ===============================================
    # # make graph
    fig = plt.figure(figsize=(17, 10))
    fig.subplots_adjust(wspace=0.4, left=0.05, right=0.95, hspace=0.4)
    gs = fig.add_gridspec(4, 8)
    plt.ion()
    cb = None

    # create axes
    axes = {}
    axes["xuv_Et"] = fig.add_subplot(gs[0:2, 0:2])
    axes["xuv_It"] = fig.add_subplot(gs[2:, 0:2])
    axes["xuv_f"] = fig.add_subplot(gs[0:2, 2:4])
    axes["xuv_f_phase"] = axes["xuv_f"].twinx()
    axes["trace_A2"] = fig.add_subplot(gs[0:2, 4:6])
    axes["trace_A"] = fig.add_subplot(gs[0:2, 6:])
    axes["trace_diff"] = fig.add_subplot(gs[2:4, 5:7])

    # calculate feed dicts
    feed_dicts = []
    step = 0.4
    gdd_vals = np.arange(-3.0, 3.0 + step, step)
    for val in gdd_vals:
        feed_dict_i = {
            xuv_coefs_in: np.array([[0.0, val, 0.0, 0.0, 0.0]]),
            ir_values_in: np.array([[1.0, 0.0, 0.0, 0.0]])
        }
        feed_dicts.append(feed_dict_i)

    gif_images = []

    for feed_dict in feed_dicts:
        # generate output
        xuv_out = sess.run(xuv_E_prop, feed_dict=feed_dict)
        out_A = sess.run(image2, feed_dict=feed_dict)
        out_A2 = sess.run(image2_2, feed_dict=feed_dict)
        # plot output
        axes["xuv_Et"].cla()
        axes["xuv_Et"].plot(np.real(xuv_out["t"][0]), color="blue")
        axes["xuv_Et"].set_title("$E(t)$")
        axes["xuv_Et"].set_xticks([])
        axes["xuv_Et"].set_ylim(-0.05, 0.05)

        axes["xuv_It"].cla()
        xuv_time_fs = xuv_spectrum.spectrum.tmat * sc.physical_constants['atomic unit of time'][0] * 1e18
        I_t = np.abs(xuv_out["t"][0]) ** 2
        axes["xuv_It"].plot(xuv_time_fs,
                            I_t, color="black")

        # calc fwhm
        halfmaxI = np.max(I_t) / 2
        I2_I = np.abs(I_t - halfmaxI)
        sorted = np.argsort(I2_I)
        index1 = sorted[0]
        index2 = find_second_minima(sorted, index1)
        fwhm = np.abs(xuv_time_fs[index1] - xuv_time_fs[index2])
        axes["xuv_It"].plot([xuv_time_fs[index1], xuv_time_fs[index2]], [halfmaxI, halfmaxI], color="red",
                            linewidth=2)
        axes["xuv_It"].text(0.7, 0.8, "FWHM [as]: " + str(round(fwhm, 2)), transform=axes["xuv_It"].transAxes,
                            color="red",
                            backgroundcolor="white")
        axes["xuv_It"].set_xlabel("time [as]")
        axes["xuv_It"].set_title("$I(t)$")
        axes["xuv_It"].set_ylim(0, 0.002)

        axes["xuv_f"].cla()
        xuv_f_hz = xuv_spectrum.spectrum.fmat_cropped / sc.physical_constants['atomic unit of time'][0]
        xuv_f_hz = xuv_f_hz * 1e-17
        axes["xuv_f"].plot(xuv_f_hz, np.abs(xuv_out["f_cropped"][0]) ** 2, color="black")
        axes["xuv_f"].set_title("XUV spectral phase")
        axes["xuv_f"].set_xlabel("frequency [$10^{17}$Hz]")
        axes["xuv_f_phase"].cla()
        axes["xuv_f_phase"].plot(xuv_f_hz, xuv_out["phasecurve_cropped"][0], color="green")
        axes["xuv_f_phase"].tick_params(axis='y', colors='green')
        axes["xuv_f_phase"].set_ylim(-40, 40)

        axes["trace_A2"].cla()
        axes["trace_A2"].pcolormesh(
            phase_parameters.params.delay_values * sc.physical_constants['atomic unit of time'][0] * 1e15,
            phase_parameters.params.K,
            out_A2, cmap="jet")
        axes["trace_A2"].set_xlabel("Delay [fs]")
        axes["trace_A2"].set_ylabel("Energy [eV]")
        axes["trace_A2"].set_title(r"$\int \frac{1}{2} A(t)^2_L$")
        axes["trace_A2"].set_yticks([])

        axes["trace_A"].cla()
        axes["trace_A"].pcolormesh(
            phase_parameters.params.delay_values * sc.physical_constants['atomic unit of time'][0] * 1e15,
            phase_parameters.params.K,
            out_A, cmap="jet")
        axes["trace_A"].set_xlabel("Delay [fs]")
        axes["trace_A"].set_ylabel("Energy [eV]")
        axes["trace_A"].set_title(r"without $\int \frac{1}{2} A(t)^2_L$(before)")

        axes["trace_diff"].cla()
        diff_im = np.abs(out_A - out_A2)
        # for setting the colorbar
        diff_im[0, 0] = 0.2
        diff_im[0, 1] = 0.0
        im = axes["trace_diff"].pcolormesh(
            phase_parameters.params.delay_values * sc.physical_constants['atomic unit of time'][0] * 1e15,
            phase_parameters.params.K,
            diff_im, cmap="jet")
        if cb is None:
            cb = fig.colorbar(im, ax=axes["trace_diff"])

        axes["trace_diff"].set_xlabel("Delay [fs]")
        axes["trace_diff"].set_ylabel("Energy [eV]")
        axes["trace_diff"].set_title("$|Trace_1 - Trace_2|$")

        textdraw = "_XUV Phase_"
        for type, phasecoef in zip(["1", "2", "3", "4", "5"], feed_dict[xuv_coefs_in][0]):
            textdraw += "\n" + "$\phi$" + type + " : " + '%.2f' % phasecoef
        axes["xuv_f_phase"].text(0.5, -1.0, textdraw, ha="center", transform=axes["xuv_f_phase"].transAxes)

        plt.pause(0.001)
        fig.canvas.draw()
        image_draw = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image_draw = image_draw.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        gif_images.append(image_draw)

    print("making gif")
    imageio.mimsave('./A2diff2.gif', gif_images, fps=10)


def find_second_minima(sorted, index1):
    # function for finding the second minima in fwhm calculation
    for j, i in enumerate(sorted[1:]):
        if np.abs(i - index1) > j + 1:
            return i


def autocorrelate(trace):

    correlate = tf.expand_dims(trace, axis=1) * tf.expand_dims(trace, axis=2)
    summation = tf.reduce_sum(correlate, axis=0)
    return summation


def proof_trace(trace):
    freq = tf_fft(tensor=tf.complex(real=trace, imag=tf.zeros_like(trace)),
                               shift=int(len(phase_parameters.params.delay_values)/2),
                               axis=1)

    # summation along vertical axis
    summationf = tf.reduce_sum(tf.abs(freq), axis=0)

    _, top_k_ind = tf.math.top_k(summationf, k=3, sorted=True)
    # get the 2nd and 3rd max values (+/- w1)
    w1_indexes = top_k_ind[1:]

    #return w1_indexes
    w1_indexes_tens = tf.one_hot(indices=w1_indexes, depth=int(len(phase_parameters.params.delay_values)))
    w1_indexes_tens_1 = tf.reduce_sum(w1_indexes_tens, axis=0)

    filtered_f = tf.complex(real=tf.reshape(w1_indexes_tens_1, [1, -1]),
                            imag=tf.zeros_like(tf.reshape(w1_indexes_tens_1, [1, -1]))) * freq

    proof = tf.real(tf_ifft(tensor=filtered_f, shift=int(len(phase_parameters.params.delay_values) / 2),
                           axis=1))

    nodes = {}
    nodes["proof"] = proof
    nodes["freq"] = freq
    nodes["trace"] = trace
    nodes["filtered_f"] = filtered_f
    nodes["summationf"] = summationf
    nodes["w1_indexes"] = w1_indexes
    nodes["w1_indexes_tens_1"] = w1_indexes_tens_1

    return nodes


def tf_ifft(tensor, shift, axis=0):

    shifted = tf.manip.roll(tensor, shift=shift, axis=axis)
    # fft
    time_domain_not_shifted = tf.ifft(shifted)
    # shift again
    time_domain = tf.manip.roll(time_domain_not_shifted, shift=shift, axis=axis)

    return time_domain


def tf_fft(tensor, shift, axis=0):

    shifted = tf.manip.roll(tensor, shift=shift, axis=axis)
    # fft
    freq_domain_not_shifted = tf.fft(shifted)
    # shift again
    freq_domain = tf.manip.roll(freq_domain_not_shifted, shift=shift, axis=axis)

    return freq_domain


def xuv_taylor_to_E(coefficients_in):

    assert int(coefficients_in.shape[1]) == phase_parameters.params.xuv_phase_coefs

    amplitude = phase_parameters.params.amplitude

    Ef = tf.constant(xuv_spectrum.spectrum.Ef, dtype=tf.complex64)
    Ef = tf.reshape(Ef, [1, -1])
    Ef_photon = tf.constant(xuv_spectrum.spectrum.Ef_photon, dtype=tf.complex64)
    Ef_photon = tf.reshape(Ef_photon, [1, -1])

    fmat_taylor = tf.constant(xuv_spectrum.spectrum.fmat-xuv_spectrum.spectrum.f0, dtype=tf.float32)

    # create factorials
    factorials = tf.constant(factorial(np.array(range(coefficients_in.shape[1]))+1), dtype=tf.float32)
    factorials = tf.reshape(factorials, [1, -1, 1])

    # create exponents
    exponents = tf.constant(np.array(range(coefficients_in.shape[1]))+1, dtype=tf.float32)

    # reshape the taylor fmat
    fmat_taylor = tf.reshape(fmat_taylor, [1, 1, -1])

    # reshape the exponential matrix
    exp_mat = tf.reshape(exponents, [1, -1, 1])

    # raise the fmat to the exponential power
    exp_mat_fmat = tf.pow(fmat_taylor, exp_mat)

    # scale the coefficients
    amplitude_mat = tf.constant(amplitude, dtype=tf.float32)
    amplitude_mat = tf.reshape(amplitude_mat, [1, -1, 1])

    # amplitude scales with exponent
    amplitude_scaler = tf.pow(amplitude_mat, exp_mat)

    # additional scaler
    # these are arbitrary numbers that were found to keep the field in the time window


    # for sample 2
    # scaler_2 = tf.constant(np.array([1.0, 1.0, 0.2, 0.06, 0.04]).reshape(1,-1,1), dtype=tf.float32)

    # for sample 3
    # ++++ force the linear phase term to always be 0
    scaler_2 = tf.constant(np.array([0.0, 1.3, 0.15, 0.03, 0.01]).reshape(1,-1,1), dtype=tf.float32)



    # reshape the coef values and scale them
    coef_values = tf.reshape(coefficients_in, [tf.shape(coefficients_in)[0], -1, 1]) * amplitude_scaler * scaler_2

    # divide by the factorials
    coef_div_fact = tf.divide(coef_values, factorials)

    # multiply by the fmat
    taylor_coefs_mat = coef_div_fact * exp_mat_fmat

    # this is the phase angle, summed along the taylor terms
    phasecurve = tf.reduce_sum(taylor_coefs_mat, axis=1)

    # apply the phase angle to Ef
    Ef_prop = Ef * tf.exp(tf.complex(imag=phasecurve, real=tf.zeros_like(phasecurve)))
    Ef_photon_prop = Ef_photon * tf.exp(tf.complex(imag=phasecurve, real=tf.zeros_like(phasecurve)))

    # fourier transform for time propagated signal
    Et_prop = tf_ifft(Ef_prop, shift=int(xuv_spectrum.spectrum.N/2), axis=1)
    Et_photon_prop = tf_ifft(Ef_photon_prop, shift=int(xuv_spectrum.spectrum.N/2), axis=1)

    # return the cropped E
    Ef_prop_cropped = Ef_prop[:, xuv_spectrum.spectrum.indexmin: xuv_spectrum.spectrum.indexmax]
    Ef_photon_prop_cropped = Ef_photon_prop[:, xuv_spectrum.spectrum.indexmin: xuv_spectrum.spectrum.indexmax]

    # return cropped phase curve
    phasecurve_cropped = phasecurve[:, xuv_spectrum.spectrum.indexmin: xuv_spectrum.spectrum.indexmax]

    E_prop = {}
    E_prop["f"] = Ef_prop
    E_prop["f_cropped"] = Ef_prop_cropped
    E_prop["f_photon_cropped"] = Ef_photon_prop_cropped
    E_prop["t"] = Et_prop
    E_prop["t_photon"] = Et_photon_prop
    E_prop["phasecurve_cropped"] = phasecurve_cropped
    #E_prop["coefs_divided_by_int"] = coefs_divided_by_int

    return E_prop


def ir_from_params(ir_param_values):

    amplitudes = phase_parameters.params.ir_param_amplitudes

    # construct tf nodes for middle and half range of inputs
    parameters = {}
    for key in ["phase_range", "clambda_range", "pulseduration_range", "I_range"]:
        parameters[key] = {}

        # get the middle and half the range of the variables
        parameters[key]["avg"] = (amplitudes[key][0] + amplitudes[key][1])/2
        parameters[key]["half_range"] = (amplitudes[key][1] - amplitudes[key][0]) / 2

        # create tensorflow constants
        parameters[key]["tf_avg"] = tf.constant(parameters[key]["avg"], dtype=tf.float32)
        parameters[key]["tf_half_range"] = tf.constant(parameters[key]["half_range"], dtype=tf.float32)


    # construct param values from normalized input
    scaled_tf_values = {}

    for i, key in enumerate(["phase_range", "clambda_range", "pulseduration_range", "I_range"]):
        scaled_tf_values[key.split("_")[0]] = parameters[key]["tf_avg"] + ir_param_values[:, i] * parameters[key]["tf_half_range"]

    # convert to SI units
    W = 1
    cm = 1e-2
    um = 1e-6
    fs = 1e-15
    atts = 1e-18

    scaled_tf_values_si = {}
    scaled_tf_values_si["I"] = scaled_tf_values["I"] * 1e13 * W / cm ** 2
    scaled_tf_values_si["f0"] = sc.c / (um * scaled_tf_values["clambda"])
    scaled_tf_values_si["t0"] =  scaled_tf_values["pulseduration"] * fs

    # calculate ponderomotive energy in SI units
    Up = (sc.elementary_charge ** 2 * tf.abs(scaled_tf_values_si["I"])) / (2 * sc.c * sc.epsilon_0 * sc.electron_mass * (2 * np.pi * scaled_tf_values_si["f0"]) ** 2)

    # convert to AU
    values_au = {}
    values_au["Up"] = Up / sc.physical_constants['atomic unit of energy'][0]
    values_au["f0"] = scaled_tf_values_si["f0"] * sc.physical_constants['atomic unit of time'][0]
    values_au["t0"] = scaled_tf_values_si["t0"] / sc.physical_constants['atomic unit of time'][0]

    # calculate driving amplitude in AU
    E0 = tf.sqrt(4 * values_au["Up"] * (2 * np.pi * values_au["f0"]) ** 2)

    # set up the driving IR field amplitude in AU
    tf_tmat = tf.reshape(tf.constant(ir_spectrum.ir_spectrum.tmat, dtype=tf.float32), [1, -1])
    # tf_fmat = tf.reshape(tf.constant(ir_spectrum.ir_spectrum.fmat, dtype=tf.float32), [1, -1])

    # slow oscilating envelope
    Et_slow_osc = tf.reshape(E0, [-1, 1]) * tf.exp(-2*np.log(2) * (tf_tmat / tf.reshape(values_au["t0"], [-1, 1]))**2)

    # fast oscilating envelope
    phase = 2 * np.pi * tf.reshape(values_au["f0"], [-1, 1]) * tf_tmat
    Et_fast_osc = tf.exp(tf.complex(imag=phase, real=tf.zeros_like(phase)))

    # Pulse before phase applied
    Et = tf.complex(real=Et_slow_osc, imag=tf.zeros_like(Et_slow_osc)) * Et_fast_osc

    # Fourier transform
    Ef = tf_fft(Et, shift=int(len(ir_spectrum.ir_spectrum.tmat)/2), axis=1)

    # apply phase angle
    phase = tf.reshape(scaled_tf_values["phase"], [-1, 1])
    Ef_phase = Ef * tf.exp(tf.complex(imag=phase, real=tf.zeros_like(phase)))

    # inverse fourier transform
    Et_phase = tf_ifft(Ef_phase, shift=int(len(ir_spectrum.ir_spectrum.tmat) / 2), axis=1)

    # crop the phase
    Ef_phase_cropped = Ef_phase[:, ir_spectrum.ir_spectrum.start_index:ir_spectrum.ir_spectrum.end_index]

    E_prop = {}
    E_prop["f"] = Ef_phase
    E_prop["f_cropped"] = Ef_phase_cropped
    E_prop["t"] = Et_phase

    out = {}
    out["scaled_values"] = scaled_tf_values
    out["E_prop"] = E_prop

    return out


def streaking_trace_old(xuv_cropped_f_in, ir_cropped_f_in):

    Ip = phase_parameters.params.Ip

    global p
    global tau_index
    global tau_values
    global p_values
    global k_values
    global padded_xuv_f
    global xuv_time_domain
    global padded_ir_f
    global ir_time_domain
    global eV_values




    # define constants
    # xuv_fmat = tf.constant(xuv.fmat, dtype=tf.float32)
    # ir_fmat = tf.constant(ir.fmat, dtype=tf.float32)

    # zero pad the spectrum of ir and xuv input to match the full fmat
    # [pad_before , padafter]
    paddings_xuv = tf.constant([[xuv_spectrum.spectrum.indexmin, xuv_spectrum.spectrum.N - xuv_spectrum.spectrum.indexmax]], dtype=tf.int32)
    padded_xuv_f = tf.pad(xuv_cropped_f_in, paddings_xuv)

    # same for the IR
    paddings_ir = tf.constant([[ir_spectrum.ir_spectrum.start_index, ir_spectrum.ir_spectrum.N - ir_spectrum.ir_spectrum.end_index]], dtype=tf.int32)
    padded_ir_f = tf.pad(ir_cropped_f_in, paddings_ir)

    # fourier transform the padded xuv
    xuv_time_domain = tf_ifft(tensor=padded_xuv_f, shift=int(xuv_spectrum.spectrum.N / 2))

    # fourier transform the padded ir
    ir_time_domain = tf_ifft(tensor=padded_ir_f, shift=int(ir_spectrum.ir_spectrum.N / 2))

    # zero pad the ir in frequency space to match dt of xuv
    assert (1 / (ir_spectrum.ir_spectrum.df * xuv_spectrum.spectrum.dt)) - math.ceil((1 / (ir_spectrum.ir_spectrum.df * xuv_spectrum.spectrum.dt))) < 0.000000001
    N_new = math.ceil(1 / (ir_spectrum.ir_spectrum.df * xuv_spectrum.spectrum.dt))
    f_pad_2 = ir_spectrum.ir_spectrum.df * np.arange(-N_new / 2, N_new / 2, 1)
    t_pad_2 = xuv_spectrum.spectrum.dt * np.arange(-N_new / 2, N_new / 2, 1)
    N_current = ir_spectrum.ir_spectrum.N
    pad_2 = (N_new - N_current) / 2
    assert int(pad_2) - pad_2 == 0
    paddings_ir_2 = tf.constant([[int(pad_2), int(pad_2)]], dtype=tf.int32)
    padded_ir_2 = tf.pad(padded_ir_f, paddings_ir_2)

    # calculate ir with matching dt in time
    ir_t_matched_dt = tf_ifft(tensor=padded_ir_2, shift=int(N_new / 2))

    # match the scale of the original
    scale_factor = tf.constant(N_new / ir_spectrum.ir_spectrum.N, dtype=tf.complex64)

    ir_t_matched_dt_scaled = ir_t_matched_dt * scale_factor

    # integrate ir pulse
    A_t = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(tf.real(ir_t_matched_dt_scaled))
    flipped1 = tf.reverse(A_t, axis=[0])
    flipped_integral = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(flipped1, axis=0)
    A_t_integ_t_phase = tf.reverse(flipped_integral, axis=[0])

    # find middle index point
    middle = int(N_new / 2)
    rangevals = np.array(range(xuv_spectrum.spectrum.N)) - xuv_spectrum.spectrum.N / 2
    middle_indexes = np.array([middle] * xuv_spectrum.spectrum.N) + rangevals

    # maximum add to zero before would be out of bounds
    max_steps = int(N_new / 2 - xuv_spectrum.spectrum.N / 2)

    # use this dt to scale the image size along tau axis
    #dtau_index = 84  # to match measured
    dtau_index = 180 # to match measured
    # dtau_index = 75

    N_tau = int(max_steps / dtau_index)


    if N_tau % 2 != 0:
        N_tau += -1

    N_tau = 29

    tau_index = dtau_index * np.arange(-N_tau, N_tau, 1, dtype=int)

    # Number of points must be even
    # assert N_tau % 2 == 0
    # assert type(dtau_index) == int
    # assert abs(tau_index[0]) < max_steps

    indexes = middle_indexes.reshape(-1, 1) + tau_index.reshape(1, -1)
    tau_values = tau_index * xuv_spectrum.spectrum.dt  # atomic units

    #print(tau_values*sc.physical_constants['atomic unit of time'][0])
    #exit(0)

    # gather values from integrated array
    ir_values = tf.gather(A_t_integ_t_phase, indexes.astype(np.int))
    ir_values = tf.expand_dims(ir_values, axis=0)

    # create momentum vector
    #p = np.linspace(3, 6.5, 200).reshape(-1, 1, 1) # previously
    #p = np.linspace(1.917, 5.0719, 200).reshape(-1, 1, 1)
    # p = np.linspace(1.8, 5.5, 235).reshape(-1, 1, 1)
    # p_values = np.squeeze(p)  # atomic units
    # K = (0.5 * p ** 2)

    # set dK = 1eV
    K = np.arange(50, 351, 1) # eV
    eV_values = np.array(K)
    # convert K to atomic units
    K = K * sc.electron_volt  # joules
    K = K / sc.physical_constants['atomic unit of energy'][0]  # a.u.
    K = K.reshape(-1, 1, 1)
    p = np.sqrt(2 * K).reshape(-1, 1, 1)
    k_values = np.squeeze(K) # atomic untis
    p_values = np.squeeze(p)  # atomic units

    # convert to tensorflow
    p_tf = tf.constant(p, dtype=tf.float32)
    K_tf = tf.constant(K, dtype=tf.float32)

    # 3d ir mat
    p_A_t_integ_t_phase3d = p_tf * ir_values
    ir_phi = tf.exp(tf.complex(imag=(p_A_t_integ_t_phase3d), real=tf.zeros_like(p_A_t_integ_t_phase3d)))

    # add fourier transform term
    e_fft = np.exp(-1j * (K + Ip) * xuv_spectrum.spectrum.tmat.reshape(1, -1, 1))
    e_fft_tf = tf.constant(e_fft, dtype=tf.complex64)

    # add xuv to integrate over
    xuv_time_domain_integrate = tf.reshape(xuv_time_domain, [1, -1, 1])

    # multiply elements together
    product = xuv_time_domain_integrate * ir_phi * e_fft_tf

    # integrate over the xuv time
    integration = tf.constant(xuv_spectrum.spectrum.dt, dtype=tf.complex64) * tf.reduce_sum(product, axis=1)

    # absolute square the matrix
    image_not_scaled = tf.square(tf.abs(integration))

    scaled = image_not_scaled - tf.reduce_min(image_not_scaled)
    image = scaled / tf.reduce_max(scaled)

    parameters = {}
    parameters["p"] = p
    parameters["tau_index"] = tau_index
    parameters["tau_values"] = tau_values
    parameters["p_values"] = p_values
    parameters["k_values"] = k_values
    parameters["padded_xuv_f"] = padded_xuv_f
    parameters["xuv_time_domain"] = xuv_time_domain
    parameters["padded_ir_f"] = padded_ir_f
    parameters["ir_time_domain"] = ir_time_domain
    parameters["eV_values"] = eV_values

    return image, parameters


def streaking_traceA(xuv_cropped_f_in, ir_cropped_f_in):

    # this is the second version of streaking trace generator which
    # accepts pre set delay values



    # ionization potential
    Ip = phase_parameters.params.Ip

    #-----------------------------------------------------------------
    # zero pad the spectrum of ir and xuv input to match the full original f matrices
    #-----------------------------------------------------------------
    # [pad_before , padafter]
    paddings_xuv = tf.constant(
        [[xuv_spectrum.spectrum.indexmin, xuv_spectrum.spectrum.N - xuv_spectrum.spectrum.indexmax]], dtype=tf.int32)
    padded_xuv_f = tf.pad(xuv_cropped_f_in, paddings_xuv)
    # same for the IR
    paddings_ir = tf.constant(
        [[ir_spectrum.ir_spectrum.start_index, ir_spectrum.ir_spectrum.N - ir_spectrum.ir_spectrum.end_index]],
        dtype=tf.int32)
    padded_ir_f = tf.pad(ir_cropped_f_in, paddings_ir)
    # fourier transform the padded xuv
    xuv_time_domain = tf_ifft(tensor=padded_xuv_f, shift=int(xuv_spectrum.spectrum.N / 2))
    # fourier transform the padded ir
    ir_time_domain = tf_ifft(tensor=padded_ir_f, shift=int(ir_spectrum.ir_spectrum.N / 2))


    #------------------------------------------------------------------
    #------ zero pad ir in frequency space to match xuv timestep-------
    #------------------------------------------------------------------
    # calculate N required to match timestep
    N_req = int(1 / (xuv_spectrum.spectrum.dt * ir_spectrum.ir_spectrum.df))
    # this much needs to be padded to each side
    pad_2 = int((N_req - ir_spectrum.ir_spectrum.N) / 2)
    # pad the IR to match dt of xuv
    paddings_ir_2 = tf.constant([[pad_2, pad_2]], dtype=tf.int32)
    padded_ir_2 = tf.pad(padded_ir_f, paddings_ir_2)
    # calculate ir with matching dt in time
    ir_t_matched_dt = tf_ifft(tensor=padded_ir_2, shift=int(N_req / 2))
    # match the scale of the original
    scale_factor = tf.constant(N_req/ ir_spectrum.ir_spectrum.N, dtype=tf.complex64)
    ir_t_matched_dt_scaled = ir_t_matched_dt * scale_factor


    #------------------------------------------------------------------
    # ---------------------integrate ir pulse--------------------------
    #------------------------------------------------------------------
    A_t = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(tf.real(ir_t_matched_dt_scaled))
    flipped1 = tf.reverse(A_t, axis=[0])
    flipped_integral = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(flipped1, axis=0)
    A_t_integ_t_phase = tf.reverse(flipped_integral, axis=[0])


    # ------------------------------------------------------------------
    # ---------------------make ir t axis-------------------------------
    # ------------------------------------------------------------------
    ir_taxis = xuv_spectrum.spectrum.dt * np.arange(-N_req/2, N_req/2, 1)



    # ------------------------------------------------------------------
    # ---------------------find indexes of tau values-------------------
    # ------------------------------------------------------------------
    center_indexes = []
    for delay_value in phase_parameters.params.delay_values:
        index = np.argmin(np.abs(delay_value - ir_taxis))
        center_indexes.append(index)
    center_indexes = np.array(center_indexes)
    rangevals = np.array(range(xuv_spectrum.spectrum.N)) - int((xuv_spectrum.spectrum.N/2))
    delayindexes = center_indexes.reshape(1, -1) + rangevals.reshape(-1, 1)


    # ------------------------------------------------------------------
    # ------------gather values from integrated array-------------------
    # ------------------------------------------------------------------
    ir_values = tf.gather(A_t_integ_t_phase, delayindexes.astype(np.int))
    ir_values = tf.expand_dims(ir_values, axis=0)



    #------------------------------------------------------------------
    #-------------------construct streaking trace----------------------
    #------------------------------------------------------------------
    # convert K to atomic units
    K = phase_parameters.params.K * sc.electron_volt  # joules
    K = K / sc.physical_constants['atomic unit of energy'][0]  # a.u.
    K = K.reshape(-1, 1, 1)
    p = np.sqrt(2 * K).reshape(-1, 1, 1)
    # convert to tensorflow
    p_tf = tf.constant(p, dtype=tf.float32)
    # 3d ir mat
    p_A_t_integ_t_phase3d = p_tf * ir_values
    ir_phi = tf.exp(tf.complex(imag=(p_A_t_integ_t_phase3d), real=tf.zeros_like(p_A_t_integ_t_phase3d)))
    # add fourier transform term
    e_fft = np.exp(-1j * (K + Ip) * xuv_spectrum.spectrum.tmat.reshape(1, -1, 1))
    e_fft_tf = tf.constant(e_fft, dtype=tf.complex64)
    # add xuv to integrate over
    xuv_time_domain_integrate = tf.reshape(xuv_time_domain, [1, -1, 1])
    # multiply elements together
    product = xuv_time_domain_integrate * ir_phi * e_fft_tf
    # integrate over the xuv time
    integration = tf.constant(xuv_spectrum.spectrum.dt, dtype=tf.complex64) * tf.reduce_sum(product, axis=1)
    # absolute square the matrix
    image_not_scaled = tf.square(tf.abs(integration))
    scaled = image_not_scaled - tf.reduce_min(image_not_scaled)
    image = scaled / tf.reduce_max(scaled)

    return image


def streaking_trace(xuv_cropped_f_in, ir_cropped_f_in):


    # define the angle for streaking trace collection
    theta_max = np.pi/2
    N_theta = 5
    angle_in = tf.constant(np.linspace(0, theta_max, N_theta), dtype=tf.float32)
    Beta_in =  1


    # this is the second version of streaking trace generator which also includes
    # the A^2 term in the integral

    # ionization potential
    Ip = phase_parameters.params.Ip

    #-----------------------------------------------------------------
    # zero pad the spectrum of ir and xuv input to match the full original f matrices
    #-----------------------------------------------------------------
    # [pad_before , padafter]
    paddings_xuv = tf.constant(
        [[xuv_spectrum.spectrum.indexmin, xuv_spectrum.spectrum.N - xuv_spectrum.spectrum.indexmax]], dtype=tf.int32)
    padded_xuv_f = tf.pad(xuv_cropped_f_in, paddings_xuv)
    # same for the IR
    paddings_ir = tf.constant(
        [[ir_spectrum.ir_spectrum.start_index, ir_spectrum.ir_spectrum.N - ir_spectrum.ir_spectrum.end_index]],
        dtype=tf.int32)
    padded_ir_f = tf.pad(ir_cropped_f_in, paddings_ir)
    # fourier transform the padded xuv
    xuv_time_domain = tf_ifft(tensor=padded_xuv_f, shift=int(xuv_spectrum.spectrum.N / 2))
    # fourier transform the padded ir
    ir_time_domain = tf_ifft(tensor=padded_ir_f, shift=int(ir_spectrum.ir_spectrum.N / 2))


    #------------------------------------------------------------------
    #------ zero pad ir in frequency space to match xuv timestep-------
    #------------------------------------------------------------------
    # calculate N required to match timestep
    N_req = int(1 / (xuv_spectrum.spectrum.dt * ir_spectrum.ir_spectrum.df))
    # this much needs to be padded to each side
    pad_2 = int((N_req - ir_spectrum.ir_spectrum.N) / 2)
    # pad the IR to match dt of xuv
    paddings_ir_2 = tf.constant([[pad_2, pad_2]], dtype=tf.int32)
    padded_ir_2 = tf.pad(padded_ir_f, paddings_ir_2)
    # calculate ir with matching dt in time
    ir_t_matched_dt = tf_ifft(tensor=padded_ir_2, shift=int(N_req / 2))
    # match the scale of the original
    scale_factor = tf.constant(N_req/ ir_spectrum.ir_spectrum.N, dtype=tf.complex64)
    ir_t_matched_dt_scaled = ir_t_matched_dt * scale_factor


    #------------------------------------------------------------------
    # ---------------------integrate ir pulse--------------------------
    #------------------------------------------------------------------
    A_t = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(tf.real(ir_t_matched_dt_scaled))

    # integrate A_L(t)
    flipped1 = tf.reverse(A_t, axis=[0])
    flipped_integral = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(flipped1, axis=0)
    A_t_integ_t_phase = tf.reverse(flipped_integral, axis=[0])

    # integrate A_L(t)^2
    flipped1_2 = tf.reverse(A_t**2, axis=[0])
    flipped_integral_2 = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(flipped1_2, axis=0)
    A_t_integ_t_phase_2 = tf.reverse(flipped_integral_2, axis=[0])



    # ------------------------------------------------------------------
    # ---------------------make ir t axis-------------------------------
    # ------------------------------------------------------------------
    ir_taxis = xuv_spectrum.spectrum.dt * np.arange(-N_req/2, N_req/2, 1)



    # ------------------------------------------------------------------
    # ---------------------find indexes of tau values-------------------
    # ------------------------------------------------------------------
    center_indexes = []
    delay_vals_au = phase_parameters.params.delay_values/sc.physical_constants['atomic unit of time'][0]
    for delay_value in delay_vals_au:
        index = np.argmin(np.abs(delay_value - ir_taxis))
        center_indexes.append(index)
    center_indexes = np.array(center_indexes)
    rangevals = np.array(range(xuv_spectrum.spectrum.N)) - int((xuv_spectrum.spectrum.N/2))
    delayindexes = center_indexes.reshape(1, -1) + rangevals.reshape(-1, 1)


    # ------------------------------------------------------------------
    # ------------gather values from integrated array-------------------
    # ------------------------------------------------------------------
    ir_values = tf.gather(A_t_integ_t_phase, delayindexes.astype(np.int))
    A_t_values = tf.gather(A_t, delayindexes.astype(np.int))
    ir_values = tf.expand_dims(tf.expand_dims(ir_values, axis=0), axis=3)
    A_t_values = tf.expand_dims(tf.expand_dims(A_t_values, axis=0), axis=3)
    # for the squared integral
    ir_values_2 = tf.gather(A_t_integ_t_phase_2, delayindexes.astype(np.int))
    ir_values_2 = tf.expand_dims(tf.expand_dims(ir_values_2, axis=0), axis=3)



    #------------------------------------------------------------------
    #-------------------construct streaking trace----------------------
    #------------------------------------------------------------------
    # convert K to atomic units
    K = phase_parameters.params.K * sc.electron_volt  # joules
    K = K / sc.physical_constants['atomic unit of energy'][0]  # a.u.
    K = K.reshape(-1, 1, 1, 1)
    p = np.sqrt(2 * K).reshape(-1, 1, 1, 1)
    # theta_max = np.pi # 90 degrees
    # angle_in = np.linspace(0, theta_max, 10)

    spec_angle = tf.reshape(tf.cos(angle_in), [1, 1, 1, -1])
    # convert to tensorflow
    p_tf = tf.constant(p, dtype=tf.float32)

    # test
    # xuv_coefs = tf.placeholder(tf.float32, shape=[None, 5])
    # ir_values_in = tf.placeholder(tf.float32, shape=[None, 4])
    # gen_xuv = xuv_taylor_to_E(xuv_coefs)
    # ir_E_prop = ir_from_params(ir_values_in)
    # with tf.Session() as sess:
    #     # feed dict to get xuv and ir output
    #     feed_dict = {ir_values_in:np.array([[0.0, 0.0, 1.0, 0.0]]), xuv_coefs:np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])}
    #     xuv_cropped_out = sess.run(gen_xuv["f_cropped"], feed_dict=feed_dict)
    #     ir_cropped_out = sess.run(ir_E_prop["f_cropped"], feed_dict=feed_dict)
    #     feed_dict = {xuv_cropped_f_in:xuv_cropped_out[0] , ir_cropped_f_in:ir_cropped_out[0]}
    #     ir_values_out = sess.run(ir_values, feed_dict=feed_dict)

    p_A_t_integ_t_phase3d = spec_angle * p_tf * ir_values + 0.5 * ir_values_2
    ir_phi = tf.exp(tf.complex(imag=(p_A_t_integ_t_phase3d), real=tf.zeros_like(p_A_t_integ_t_phase3d)))
    # add fourier transform term
    e_fft = np.exp(-1j * (K + Ip) * xuv_spectrum.spectrum.tmat.reshape(1, -1, 1, 1))
    e_fft_tf = tf.constant(e_fft, dtype=tf.complex64)
    # add xuv to integrate over
    xuv_time_domain_integrate = tf.reshape(xuv_time_domain, [1, -1, 1, 1])
    # multiply elements together

    # axes:
    # (301, 2048, 98)
    # (K, xuv_time, tau_delay)
    # --> expand dimmension for angle -->
    # (301, 2048, 98, ??)
    # (K, xuv_time, tau_delay, angle)
    # angular distribution term calculated from equation
    angular_distribution = 1 + (Beta_in / 2)  * (3 * (tf.cos(angle_in))**2 - 1)
    angular_distribution = tf.reshape(angular_distribution, [1, 1, 1, -1])
    angular_distribution = tf.complex(imag=tf.zeros_like(angular_distribution), real=angular_distribution)

    # A_t_values
    # p_tf
    alpha = 2*Ip
    # dipole matrix element
    dipole_p = p_tf
    dipole_mat = (2**(7 / 2) * alpha**(5 / 4)) / (np.pi)
    dipole_mat = dipole_mat * ((dipole_p) / ((dipole_p**2 + alpha)**3))
    dipole_mat = tf.complex(imag=dipole_mat, real=tf.zeros_like(dipole_mat))

    product = angular_distribution * xuv_time_domain_integrate * dipole_mat * ir_phi * e_fft_tf
    # integrate over the xuv time
    integration = tf.constant(xuv_spectrum.spectrum.dt, dtype=tf.complex64) * tf.reduce_sum(product, axis=1)
    # absolute square the matrix
    image_not_scaled = tf.square(tf.abs(integration))
    image_not_scaled = image_not_scaled * tf.reshape(tf.sin(angle_in), [1, 1, -1])

    # integrate along the theta axis
    dtheta = angle_in[1] - angle_in[0]
    theta_integration = dtheta * tf.reduce_sum(image_not_scaled, axis=2)

    scaled = theta_integration - tf.reduce_min(theta_integration)
    image = scaled / tf.reduce_max(scaled)

    out = {}
    out["image"] = image
    out["A_t_values"] = A_t_values
    out["p_tf"] = p_tf
    out["dipole_mat"] = dipole_mat
    out["alpha"] = alpha
    out["K"] = K
    return out



def streaking_trace_no_angle(xuv_cropped_f_in, ir_cropped_f_in):
    # this is the second version of streaking trace generator which also includes
    # the A^2 term in the integral

    # ionization potential
    Ip = phase_parameters.params.Ip

    #-----------------------------------------------------------------
    # zero pad the spectrum of ir and xuv input to match the full original f matrices
    #-----------------------------------------------------------------
    # [pad_before , padafter]
    paddings_xuv = tf.constant(
        [[xuv_spectrum.spectrum.indexmin, xuv_spectrum.spectrum.N - xuv_spectrum.spectrum.indexmax]], dtype=tf.int32)
    padded_xuv_f = tf.pad(xuv_cropped_f_in, paddings_xuv)
    # same for the IR
    paddings_ir = tf.constant(
        [[ir_spectrum.ir_spectrum.start_index, ir_spectrum.ir_spectrum.N - ir_spectrum.ir_spectrum.end_index]],
        dtype=tf.int32)
    padded_ir_f = tf.pad(ir_cropped_f_in, paddings_ir)
    # fourier transform the padded xuv
    xuv_time_domain = tf_ifft(tensor=padded_xuv_f, shift=int(xuv_spectrum.spectrum.N / 2))
    # fourier transform the padded ir
    ir_time_domain = tf_ifft(tensor=padded_ir_f, shift=int(ir_spectrum.ir_spectrum.N / 2))


    #------------------------------------------------------------------
    #------ zero pad ir in frequency space to match xuv timestep-------
    #------------------------------------------------------------------
    # calculate N required to match timestep
    N_req = int(1 / (xuv_spectrum.spectrum.dt * ir_spectrum.ir_spectrum.df))
    # this much needs to be padded to each side
    pad_2 = int((N_req - ir_spectrum.ir_spectrum.N) / 2)
    # pad the IR to match dt of xuv
    paddings_ir_2 = tf.constant([[pad_2, pad_2]], dtype=tf.int32)
    padded_ir_2 = tf.pad(padded_ir_f, paddings_ir_2)
    # calculate ir with matching dt in time
    ir_t_matched_dt = tf_ifft(tensor=padded_ir_2, shift=int(N_req / 2))
    # match the scale of the original
    scale_factor = tf.constant(N_req/ ir_spectrum.ir_spectrum.N, dtype=tf.complex64)
    ir_t_matched_dt_scaled = ir_t_matched_dt * scale_factor


    #------------------------------------------------------------------
    # ---------------------integrate ir pulse--------------------------
    #------------------------------------------------------------------
    A_t = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(tf.real(ir_t_matched_dt_scaled))

    # integrate A_L(t)
    flipped1 = tf.reverse(A_t, axis=[0])
    flipped_integral = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(flipped1, axis=0)
    A_t_integ_t_phase = tf.reverse(flipped_integral, axis=[0])

    # integrate A_L(t)^2
    flipped1_2 = tf.reverse(A_t**2, axis=[0])
    flipped_integral_2 = tf.constant(-1.0 * xuv_spectrum.spectrum.dt, dtype=tf.float32) * tf.cumsum(flipped1_2, axis=0)
    A_t_integ_t_phase_2 = tf.reverse(flipped_integral_2, axis=[0])



    # ------------------------------------------------------------------
    # ---------------------make ir t axis-------------------------------
    # ------------------------------------------------------------------
    ir_taxis = xuv_spectrum.spectrum.dt * np.arange(-N_req/2, N_req/2, 1)



    # ------------------------------------------------------------------
    # ---------------------find indexes of tau values-------------------
    # ------------------------------------------------------------------
    center_indexes = []
    delay_vals_au = phase_parameters.params.delay_values/sc.physical_constants['atomic unit of time'][0]
    for delay_value in delay_vals_au:
        index = np.argmin(np.abs(delay_value - ir_taxis))
        center_indexes.append(index)
    center_indexes = np.array(center_indexes)
    rangevals = np.array(range(xuv_spectrum.spectrum.N)) - int((xuv_spectrum.spectrum.N/2))
    delayindexes = center_indexes.reshape(1, -1) + rangevals.reshape(-1, 1)


    # ------------------------------------------------------------------
    # ------------gather values from integrated array-------------------
    # ------------------------------------------------------------------
    ir_values = tf.gather(A_t_integ_t_phase, delayindexes.astype(np.int))
    ir_values = tf.expand_dims(ir_values, axis=0)
    # for the squared integral
    ir_values_2 = tf.gather(A_t_integ_t_phase_2, delayindexes.astype(np.int))
    ir_values_2 = tf.expand_dims(ir_values_2, axis=0)



    #------------------------------------------------------------------
    #-------------------construct streaking trace----------------------
    #------------------------------------------------------------------
    # convert K to atomic units
    K = phase_parameters.params.K * sc.electron_volt  # joules
    K = K / sc.physical_constants['atomic unit of energy'][0]  # a.u.
    K = K.reshape(-1, 1, 1)
    p = np.sqrt(2 * K).reshape(-1, 1, 1)
    # convert to tensorflow
    p_tf = tf.constant(p, dtype=tf.float32)
    # 3d ir mat
    p_A_t_integ_t_phase3d = p_tf * ir_values + 0.5 * ir_values_2
    ir_phi = tf.exp(tf.complex(imag=(p_A_t_integ_t_phase3d), real=tf.zeros_like(p_A_t_integ_t_phase3d)))
    # add fourier transform term
    e_fft = np.exp(-1j * (K + Ip) * xuv_spectrum.spectrum.tmat.reshape(1, -1, 1))
    e_fft_tf = tf.constant(e_fft, dtype=tf.complex64)
    # add xuv to integrate over
    xuv_time_domain_integrate = tf.reshape(xuv_time_domain, [1, -1, 1])
    # multiply elements together
    product = xuv_time_domain_integrate * ir_phi * e_fft_tf
    # integrate over the xuv time
    integration = tf.constant(xuv_spectrum.spectrum.dt, dtype=tf.complex64) * tf.reduce_sum(product, axis=1)
    # absolute square the matrix
    image_not_scaled = tf.square(tf.abs(integration))
    scaled = image_not_scaled - tf.reduce_min(image_not_scaled)
    image = scaled / tf.reduce_max(scaled)

    return image

def phase_rmse_error_test():
    # calculate transform limited trace
    # view generated xuv pulse
    xuv_coefs = tf.placeholder(tf.float32, shape=[None, 5])
    ir_values_in = tf.placeholder(tf.float32, shape=[None, 4])

    gen_xuv = xuv_taylor_to_E(xuv_coefs)
    ir_E_prop = ir_from_params(ir_values_in)["E_prop"]

    image = streaking_trace(xuv_cropped_f_in=gen_xuv["f_cropped"][0], ir_cropped_f_in=ir_E_prop["f_cropped"][0])

    feed_dict = {ir_values_in:np.array([[0.0, 0.0, 1.0, 0.0]])}
    with tf.Session() as sess:
        # feed_dict[xuv_coefs] = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
        feed_dict[xuv_coefs] = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])
        gen_trace = sess.run(image, feed_dict=feed_dict)
        xuv_out = sess.run(gen_xuv, feed_dict=feed_dict)["t"][0]

        # calculate transform limited trace
        feed_dict[xuv_coefs] = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
        tf_limited_trace = sess.run(image, feed_dict=feed_dict)
        tf_limited_xuv_out = sess.run(gen_xuv, feed_dict=feed_dict)["t"][0]

        rmse_trace = (1/len(gen_trace.reshape(-1))) * np.sum((gen_trace.reshape(-1) - tf_limited_trace.reshape(-1))**2)

        fig = plt.figure(figsize=(10,10))
        gs = fig.add_gridspec(2,2)
        ax = fig.add_subplot(gs[0,0])
        ax.pcolormesh(gen_trace, cmap="jet")
        ax.set_title("trace with dispersion")

        ax = fig.add_subplot(gs[1,0])
        ax.plot(xuv_spectrum.spectrum.tmat, np.abs(xuv_out)**2, color="black")
        ax.set_title("I(t) with dispersion")

        ax = fig.add_subplot(gs[0,1])
        ax.pcolormesh(tf_limited_trace, cmap="jet")
        ax.set_title("transform limited trace")
        ax.text(0.3, 0.7, "rmse: %.10f" % rmse_trace, transform=ax.transAxes, bbox=dict(facecolor='white'))

        ax = fig.add_subplot(gs[1,1])
        ax.plot(xuv_spectrum.spectrum.tmat, np.abs(tf_limited_xuv_out)**2, color="black")
        ax.set_title("I(t) transform limited")





        plt.show()
        exit()

        dispersion_values = np.linspace(-1.0, 1.0, 300)
        order2_rmse = []
        for order2 in dispersion_values:
            feed_dict[xuv_coefs] = np.array([[0.0, order2, 0.0, 0.0, 0.0]])
            gen_trace = sess.run(image, feed_dict=feed_dict)
            rmse_trace = (1/len(gen_trace.reshape(-1))) * np.sum((gen_trace.reshape(-1) - tf_limited_trace.reshape(-1))**2)
            order2_rmse.append(rmse_trace)
        order3_rmse = []
        for order3 in dispersion_values:
            feed_dict[xuv_coefs] = np.array([[0.0, 0.0, order3, 0.0, 0.0]])
            gen_trace = sess.run(image, feed_dict=feed_dict)
            rmse_trace = (1/len(gen_trace.reshape(-1))) * np.sum((gen_trace.reshape(-1) - tf_limited_trace.reshape(-1))**2)
            order3_rmse.append(rmse_trace)
        order4_rmse = []
        for order4 in dispersion_values:
            feed_dict[xuv_coefs] = np.array([[0.0, 0.0, 0.0, order4, 0.0]])
            gen_trace = sess.run(image, feed_dict=feed_dict)
            rmse_trace = (1/len(gen_trace.reshape(-1))) * np.sum((gen_trace.reshape(-1) - tf_limited_trace.reshape(-1))**2)
            order4_rmse.append(rmse_trace)
        order5_rmse = []
        for order5 in dispersion_values:
            feed_dict[xuv_coefs] = np.array([[0.0, 0.0, 0.0, 0.0, order5]])
            gen_trace = sess.run(image, feed_dict=feed_dict)
            rmse_trace = (1/len(gen_trace.reshape(-1))) * np.sum((gen_trace.reshape(-1) - tf_limited_trace.reshape(-1))**2)
            order5_rmse.append(rmse_trace)

        plt.figure(1)
        plt.plot(dispersion_values, order2_rmse, label="order 2 RMSE")
        plt.plot(dispersion_values, order3_rmse, label="order 3 RMSE")
        plt.plot(dispersion_values, order4_rmse, label="order 4 RMSE")
        plt.plot(dispersion_values, order5_rmse, label="order 5 RMSE")
        plt.xlabel("Normalized/scaled Dispersion Coefficient")
        plt.ylabel("RMSE compared to transform limited trace")
        # plt.yscale("log")
        plt.legend()
        # plt.show()
        plt.savefig("./dispersion_rmse.png")
        exit()


def calculate_dipole_mat(alpha, K):

    dipole_mat_np = (2**(7 / 2) * alpha**(5 / 4)) / (np.pi)
    dipole_mat_np = dipole_mat_np * ((p_tf) / ((p_tf**2 + alpha)**3))
    dipole_mat_np = 1j * dipole_mat_np
    dipole_abs_k = np.squeeze(K) * np.squeeze(np.abs(dipole_mat_np)**2)

    return dipole_abs_k


if __name__ == "__main__":
    # phase_rmse_error_test()

    # view generated xuv pulse
    xuv_coefs = tf.placeholder(tf.float32, shape=[None, 5])
    ir_values_in = tf.placeholder(tf.float32, shape=[None, 4])

    gen_xuv = xuv_taylor_to_E(xuv_coefs)
    ir_E_prop = ir_from_params(ir_values_in)["E_prop"]
    image = streaking_trace(xuv_cropped_f_in=gen_xuv["f_cropped"][0], ir_cropped_f_in=ir_E_prop["f_cropped"][0])

    feed_dict = {
            # xuv_coefs:np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
            xuv_coefs:np.array([[0.0, 1.0, 0.0, 0.0, 0.0]]),
            ir_values_in:np.array([[0.0, 0.0, 1.0, 0.0]]),
            }

    with tf.Session() as sess:
        out = sess.run(gen_xuv, feed_dict=feed_dict)
        xuv_t = out['t'][0]
        plt.figure(1)
        plt.plot(xuv_spectrum.spectrum.tmat, np.real(xuv_t), color="blue")
        plt.plot(xuv_spectrum.spectrum.tmat, np.imag(xuv_t), color="red")
        plt.plot(xuv_spectrum.spectrum.tmat, np.abs(xuv_t), color="black")
        # plot spectrogram
        out = sess.run(image["image"], feed_dict=feed_dict)
        plt.figure(2)
        plt.title("spectrogram")
        plt.pcolormesh(out, cmap="jet")
        plt.savefig("./trace_with_dipole.png")

        p_tf = sess.run(image["p_tf"], feed_dict=feed_dict)
        A_t_values = sess.run(image["A_t_values"], feed_dict=feed_dict)
        dipole_mat = sess.run(image["dipole_mat"], feed_dict=feed_dict)
        alpha = image["alpha"]
        K = image["K"]

        # calc the dipole mat without tensorflow

        # should be the same
        # plt.figure(44)
        # plt.plot(np.imag(np.squeeze(dipole_mat_np)))
        # plt.figure(45)
        # plt.plot(np.imag(np.squeeze(dipole_mat)))
        # plt.show()
        plt.figure(47)
        diple_mat_np_abs_k = calculate_dipole_mat(alpha, K)
        plt.plot(phase_parameters.params.K, diple_mat_np_abs_k)
        plt.gca().set_yscale("log")


        fig = plt.figure(figsize=(8,8))
        gs = fig.add_gridspec(2,2)
        ax = fig.add_subplot(gs[0, 0])
        fig.subplots_adjust(left=0.2, top=0.8, right=0.8, wspace=0.4, hspace=0.4)
        ax.plot(phase_parameters.params.K, diple_mat_np_abs_k)
        ax.set_xlabel("energy (eV)")
        ax.set_ylabel("$K \cdot |d(p)|^2$")
        ax.set_yscale("log")
        ax.set_title(r" momentum: $p [a.u.]$"+"\n"+r" energy: $K [a.u]$ "+"\n"+r" dipole: $d(p) = i ( \frac{2^{2/7} \alpha^{5/4}}{\pi} ) \frac{p}{(p^{2}+\alpha)^3}$ (without A(t)) "+"\n"+r" $K \cdot |d(p)|^2$")

        # plot the cross section
        with open("cross_section_ev.p", "rb") as file:
            cross_section_ev = pickle.load(file)

        ax = fig.add_subplot(gs[0, 1])
        ax.plot(phase_parameters.params.K, cross_section_ev)
        ax.set_yscale("log")
        ax.set_xlabel("energy (eV)")
        ax.set_ylabel("cross section")
        ax.set_title("cross section")

        # set the smallest point of each to be same
        cross_section_ev = set_point_to(cross_section_ev, index=100, value=0)
        diple_mat_np_abs_k = set_point_to(diple_mat_np_abs_k, index=100, value=0)


        ax = fig.add_subplot(gs[1, 1])
        ax.set_title("set both plots at energy=150 eV to 0, no log scale")
        ax.plot(phase_parameters.params.K, cross_section_ev, label="cross section")
        ax.plot(phase_parameters.params.K, diple_mat_np_abs_k, label="$K \cdot |d(p)|^2$")
        ax.legend()
        # ax.set_yscale("log")
        ax.set_xlabel("energy (eV)")

        # generate curve fitting





        plt.show()
