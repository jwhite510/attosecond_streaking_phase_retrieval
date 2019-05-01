import unsupervised_retrieval
import matplotlib.pyplot as plt
import pickle

run_name = "measured_retrieval_init"
with open("./retrieval/" + run_name + "/plot_objs.p", "rb") as file:
    plot_obj = pickle.load(file)

# import ipdb; ipdb.set_trace() # BREAKPOINT


def plot_images_fields_publication(axes, traces_meas, traces_reconstructed, xuv_f, xuv_f_phase,  xuv_f_full, xuv_t, ir_f, i,
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
            with open("./retrieval/" + run_name + "/plot_objs.p", "wb") as file:
                pickle.dump(file_objs, file)


# plot
unsupervised_retrieval.plot_images_fields(axes=plot_obj["axes"], 
                        traces_meas=plot_obj["traces_meas"],
                        traces_reconstructed=plot_obj["traces_reconstructed"], 
                        xuv_f=plot_obj["xuv_f"],
                        xuv_f_phase=plot_obj["xuv_f_phase"], 
                        xuv_f_full=plot_obj["xuv_f_full"],
                        xuv_t=plot_obj["xuv_t"], ir_f=plot_obj["ir_f"], i=plot_obj["i"],
                        run_name=plot_obj["run_name"], true_fields=plot_obj["true_fields"],
                        cost_function=plot_obj["cost_function"],
                        method=plot_obj["method"])


plt.show()




