from crab_tf import *
import tensorflow as tf
from scipy.interpolate import interp1d


class XUV_Field_rand_phase(XUV_Field):

    def __init__(self, phase_amplitude, phase_nodes, plot):

        XUV_Field.__init__(self, N=N, tmax=tmax, gdd=0, tod=0)

        middle_index = int(len(self.Et) / 2)
        lower = middle_index-int(self.span/2)
        upper = middle_index+int(self.span/2)

        # define the integration timespan of the field
        self.Et_cropped = self.Et[lower:upper]

        # define phase vector
        self.nodes = phase_amplitude * (np.random.rand(phase_nodes) - 0.5)

        axis_nodes = np.linspace(0, self.span, phase_nodes)

        axis_phase = np.array(range(self.span))

        f = interp1d(axis_nodes, self.nodes, kind='cubic')

        phase = f(axis_phase)

        # fft the pulse
        self.Et_cropped_f = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.Et_cropped)))

        # apply phase
        self.Et_cropped_f_phase = self.Et_cropped_f * np.exp(1j * phase)

        # ft back to temporal domain
        self.Et_cropped_t_phase = np.fft.fftshift((np.fft.ifft(np.fft.fftshift(self.Et_cropped_f_phase))))

        if plot:
            fig, ax = plt.subplots(4, 1, figsize=(5, 10))
            ax[0].plot(np.real(self.Et_cropped), color='blue')
            ax[0].plot(np.imag(self.Et_cropped), color='red')

            # plot the fourier transform and the phase to apply
            ax[1].plot(np.real(self.Et_cropped_f), color='blue')
            ax[1].plot(np.imag(self.Et_cropped_f), color='red')
            axtwin = ax[1].twinx()
            axtwin.plot(phase, color='green', linestyle='dashed')

            # plot the spectral pulse with phase applied
            ax[2].plot(np.real(self.Et_cropped_f_phase), color='blue')
            ax[2].plot(np.imag(self.Et_cropped_f_phase), color='red')
            axtwin = ax[2].twinx()
            axtwin.plot(np.unwrap(np.angle(self.Et_cropped_f_phase)), color='green')

            # plot the field in time domain
            ax[3].plot(np.real(self.Et_cropped_t_phase), color='blue')
            ax[3].plot(np.imag(self.Et_cropped_t_phase), color='red')

            plt.show()




xuv_test = XUV_Field(N=N, tmax=tmax, gdd=500, tod=0).Et[lower:upper]



plt.ion()

init = tf.global_variables_initializer()
with tf.Session() as sess:

    init.run()

    xuv_rand = XUV_Field_rand_phase(phase_amplitude=3, phase_nodes=120, plot=False).Et_cropped_t_phase

    strace = sess.run(image, feed_dict={xuv_input: xuv_rand.reshape(1, -1, 1)})

    plt.figure(10)
    plt.pcolormesh(tauvec, p_vec, strace, cmap='jet')

    plt.figure(11)
    plt.plot(np.real(xuv_rand), color='blue')
    plt.plot(np.imag(xuv_rand), color='red')

    plt.show()




