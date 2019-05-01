import numpy as np
import matplotlib.pyplot as plt
import pickle
from xuv_spectrum import spectrum
import phase_parameters.params as phase_params
import unsupervised_retrieval
spectrum_scaled = spectrum.fmat_hz_cropped * 1e-16


with open("noise_test10__.p", "rb") as file:
    data_retrievals = pickle.load(file)

with open("noise_test10___bootstrap.p", "rb") as file:
    data_bootstrap = pickle.load(file)



key = "unsupervised_20"
# key = "unsupervised_1895"
# key = "ga_20"
# key = "ga_1895"
run_sample = data_bootstrap[key]
initial_network_out_phase = data_retrievals["20"]["normal"]["nn_init"]["field"]["cropped_phase"]
actual_phase_curve = data_retrievals["actual_values"]["measured_trace_phase_20"]
actual_trace = data_retrievals["actual_values"]["measured_trace_20"]

# create axes
fig = plt.figure(figsize=(7,7))
gs = fig.add_gridspec(2,1)

total_samples = len(run_sample)
ax = fig.add_subplot(gs[1,0])
phase_curve = None
for thing in run_sample:

    if phase_curve is not None:
        phase_curve = np.append(phase_curve, thing["field"]["cropped_phase"].reshape(1, -1), axis=0)
    else:
        phase_curve = thing["field"]["cropped_phase"].reshape(1, -1)

# xuv spectrum / phase axis
# calculate mean
mean_vals = np.sum(phase_curve, axis=0) / np.shape(phase_curve)[0]
# calculate standard deviation
standard_dev = np.std(phase_curve, axis=0)
# plot the mean
axtwin = ax.twinx()
axtwin.plot(spectrum_scaled, mean_vals, color="green", label="Mean")


first_plotted = False
for x, this_std_val, this_mean_val in zip(spectrum_scaled[::8], standard_dev[::8], mean_vals[::8]):
    scale_fac = 1
    ymax_val = this_mean_val + (scale_fac * this_std_val/2)
    ymin_val = this_mean_val - (scale_fac * this_std_val/2)
    if not first_plotted:
        if not scale_fac == 1:
            axtwin.plot([x,x], [ymin_val, ymax_val], color="green", marker="o", 
                    label=str(scale_fac)+r"$\times$Standard Deviation")
        else:
            axtwin.plot([x,x], [ymin_val, ymax_val], color="green", marker="o", label="Standard Deviation")
        first_plotted = True
    else:
        axtwin.plot([x,x], [ymin_val, ymax_val], color="green", marker="o")

# plot the actual phase curve
axtwin.plot(spectrum_scaled, actual_phase_curve,
            color="green", label="Actual Phase", linestyle="dashed")

# plot the phase curve from the initial network output
axtwin.plot(spectrum_scaled, initial_network_out_phase,
            color="blue", label="Initial Network Output", linestyle="dashed")

axtwin.legend(loc=1)
axtwin.tick_params(axis='y', colors="green")
axtwin.set_ylabel("Phase [rad]")
axtwin.set_ylim(-150,150)

ax.plot(spectrum_scaled, np.abs(spectrum.Ef[spectrum.indexmin:spectrum.indexmax])**2,
        color="black")
ax.set_ylabel("Intensity")
ax.set_xlabel(r"$10^{16}$Hz")
ax.set_ylim(0, 1.2)


# plot the actual trace trace
ax2 = fig.add_subplot(gs[0,0])
ax2.pcolormesh(phase_params.delay_values_fs ,phase_params.K, actual_trace, cmap="jet")
ax2.set_xlabel("time [fs]")
ax2.set_ylabel("Energy [eV]")

# write the number of samples
unsupervised_retrieval.normal_text(ax2, (0.9, 1.1), "Samples = "+str(total_samples))

if key[0] == "g":
    ax2.set_title("Genetic Algorithm")
    fig.savefig("./bootstrap_ga.png")
elif key[0] == "u":
    ax2.set_title("Unsupervised Learning")
    fig.savefig("./bootstrap_unsupervised.png")

plt.show()









