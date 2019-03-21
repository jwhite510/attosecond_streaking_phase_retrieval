import pickle
import numpy as np
import matplotlib.pyplot as plt
import xuv_spectrum.spectrum as xuv_spec
import scipy.constants as sc
import imageio







with open("./unsupervised_retrieval/run1_log_b10_t1/u_fields.p", "rb") as file:
    fields = pickle.load(file=file)


xuv_time = xuv_spec.tmat * sc.physical_constants['atomic unit of time'][0] * 1e18# as
xuv_signal_time = fields["predicted_fields"]["xuv_t"]

index = 0
span = 40 # even
dt = xuv_spec.dt * sc.physical_constants['atomic unit of time'][0] # seconds

tmat_window = dt * np.arange(-span/2, span/2, 1) * 1e18# as
df = 1 / (dt * span)
fmat_window = df * np.arange(-span/2, span/2, 1)

samples = len(xuv_signal_time) - (span - 1)
#f_c_index = np.zeros((samples))

# define time values for fft windows
f_c_time_indexes = np.array(range(len(xuv_signal_time)))[(int(span/2)):-(int(span/2)-1)]
f_c_time = xuv_time[f_c_time_indexes]


f_c = np.zeros((samples, span), dtype=np.complex)



fig = plt.figure()
gs = fig.add_gridspec(2,3)
fig.subplots_adjust(hspace=0.3)

axes = {}
axes["xuv_timewindow"] = fig.add_subplot(gs[0,2])
axes["fft"] = fig.add_subplot(gs[1,:2])
axes["xuv_time_sig"] = fig.add_subplot(gs[0,:2])

plt.ion()
movie = []
while index + span -1 < len(xuv_signal_time):

    xuv_signal_time_window = xuv_signal_time[index:index+span]
    window_fft = np.fft.fftshift(np.fft.fft(xuv_signal_time_window))
    # add the fft window
    f_c[index,:] = window_fft
    # add the central index in the time window
    index_center_window = (index + (index + span)) / 2

    # plot the ffts
    axes["fft"].cla()
    axes["fft"].pcolormesh(f_c_time, fmat_window*1e-17, np.transpose(np.abs(f_c)), cmap="jet")
    axes["fft"].set_xlim(np.min(f_c_time), np.max(f_c_time))
    axes["fft"].set_xlabel("time [as]")
    axes["fft"].set_ylabel("$10^{17}$Hz")
    axes["fft"].set_ylim(0, 2)



    # plot the time slice of the xuv signal
    axes["xuv_timewindow"].cla()
    axes["xuv_timewindow"].plot(tmat_window, np.real(xuv_signal_time_window), color="blue")
    axes["xuv_timewindow"].plot([0,0], [np.min(np.real(xuv_signal_time_window)), np.max(np.real(xuv_signal_time_window))], color="red")
    axes["xuv_timewindow"].set_xlabel("time [as]")
    axes["xuv_timewindow"].set_yticks([])




    # plot the xuv signal
    axes["xuv_time_sig"].cla()
    axes["xuv_time_sig"].plot(xuv_time, np.real(xuv_signal_time), color="blue")
    axes["xuv_time_sig"].set_xlim(np.min(f_c_time), np.max(f_c_time))
    axes["xuv_time_sig"].set_xlabel("time [as]")
    axes["xuv_time_sig"].set_yticks([])
    axes["xuv_time_sig"].set_ylabel(r"real $E(t)$ $\rightarrow$")
    # plot where is the index center
    time_index_center_window = xuv_time[int(index_center_window)]
    time_index = xuv_time[index]
    time_ind_plus_span = xuv_time[index + span -1]

    axes["xuv_time_sig"].plot([time_index_center_window, time_index_center_window], [np.min(np.real(xuv_signal_time)),
                                                                           np.max(np.real(xuv_signal_time))],
                                                                            color="red")
    # plot the min and max window
    axes["xuv_time_sig"].fill_between([time_index, time_ind_plus_span],
                                      [np.max(np.real(xuv_signal_time)), np.max(np.real(xuv_signal_time))],
                                      [np.min(np.real(xuv_signal_time)), np.min(np.real(xuv_signal_time))],
                                      color="black",
                                      alpha=0.5)

    if index % 5 == 0:
        fig.canvas.draw()
        plt.pause(0.001)
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1]+(3,))
        movie.append(image)


    index += 1

plt.savefig("./completeimage.png")
imageio.mimsave("./fftwindow.gif", movie, fps=15)

