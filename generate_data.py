from crab_tf import *
import tensorflow as tf
from scipy.interpolate import interp1d
import tables
import time


class XUV_Field_rand_phase(XUV_Field):

    def __init__(self, phase_amplitude, phase_nodes, plot):

        XUV_Field.__init__(self, N=N, tmax=tmax, gdd=0, tod=0)

        middle_index = int(len(self.Et) / 2)
        lower = middle_index-int(self.span/2)
        upper = middle_index+int(self.span/2)

        # define the integration timespan of the field
        self.Et_cropped = self.Et[lower:upper]
        self.tmat_cropped = self.tmat[lower:upper]

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

        # ambiguity removal
        # set the phase angle at t0 to 0
        Envelope = self.Et_cropped_t_phase * np.exp(-2j * np.pi * self.f0 * self.tmat_cropped)
        t_0_index = np.argmin(np.abs(self.tmat_cropped - 0))
        # print('t_0_index: ', t_0_index)
        angle_at_t0 = np.angle(Envelope[t_0_index])
        Envelope_corrected = Envelope * np.exp(-1j * angle_at_t0)

        # print('np.real(Envelope[t_0_index]):', np.real(Envelope[t_0_index]))
        # print('np.imag(Envelope[t_0_index]):', np.imag(Envelope[t_0_index]))
        #
        # print('np.real(Envelope_corrected[t_0_index]):', np.real(Envelope_corrected[t_0_index]))
        # print('np.imag(Envelope_corrected[t_0_index]):', np.imag(Envelope_corrected[t_0_index]))

        Corrected_centralfreq = Envelope_corrected * np.exp(2j * np.pi * self.f0 * self.tmat_cropped)

        self.Et_cropped_t_phase = Corrected_centralfreq

        # plt.plot(np.real(Corrected_centralfreq))
        # plt.ioff()
        # plt.show()


        # Corrected_centralfreq2 = Corrected_centralfreq * np.exp(-2j * np.pi * self.f0 * self.tmat_cropped)

        # for checking angle at time 0
        # fig, ax = plt.subplots(5, 1)
        #
        # ax[0].plot(self.tmat_cropped, np.real(self.Et_cropped_t_phase), color='blue')
        # ax[0].plot(self.tmat_cropped, np.imag(self.Et_cropped_t_phase), color='red')
        #
        # ax[1].plot(self.tmat_cropped, np.real(Envelope), color='blue')
        # ax[1].plot(self.tmat_cropped, np.imag(Envelope), color='red')
        #
        # ax[2].plot(self.tmat_cropped, np.real(Envelope_corrected), color='blue')
        # ax[2].plot(self.tmat_cropped, np.imag(Envelope_corrected), color='red')
        #
        # ax[3].plot(self.tmat_cropped, np.real(Corrected_centralfreq), color='blue')
        # ax[3].plot(self.tmat_cropped, np.imag(Corrected_centralfreq), color='red')
        #
        # ax[4].plot(self.tmat_cropped, np.real(Corrected_centralfreq2), color='blue')
        # ax[4].plot(self.tmat_cropped, np.imag(Corrected_centralfreq2), color='red')
        #
        # plt.ioff()
        # plt.show()


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


# create hdf5 file
hdf5_file = tables.open_file('attstrace.hdf5', mode='w')
frog_image_f = hdf5_file.create_earray(hdf5_file.root,
                                       'trace', tables.Float16Atom(), shape=(0, len(p_vec) * len(tauvec)))
E_real_f = hdf5_file.create_earray(hdf5_file.root,
                                   'xuv_real', tables.Float16Atom(), shape=(0, len(xuv_test)))
E_imag_f = hdf5_file.create_earray(hdf5_file.root,
                                   'xuv_imag', tables.Float16Atom(), shape=(0, len(xuv_test)))
hdf5_file.close()

fig, ax = plt.subplots(2, 1)
n_samples = 20000
plt.ion()
plotting = True
init = tf.global_variables_initializer()
with tf.Session() as sess:

    init.run()

    # open the hdf5 file
    hdf5_file = tables.open_file('attstrace.hdf5', mode='a')

    for i in range(n_samples):

        # generate a random xuv pulse
        xuv_rand = XUV_Field_rand_phase(phase_amplitude=5, phase_nodes=120, plot=False).Et_cropped_t_phase

        if i % 1000 == 0:
            print('generating sample {} of {}'.format(i + 1, n_samples))
            # generate the FROG trace
            time1 = time.time()
            strace = sess.run(image, feed_dict={xuv_input: xuv_rand.reshape(1, -1, 1)})
            time2 = time.time()
            duration = time2 - time1
            print('duration: {} s'.format(round(duration, 4)))

            if plotting:
                plt.cla()
                ax[0].pcolormesh(strace, cmap='jet')
                ax[1].plot(np.abs(xuv_rand), color='black', linestyle='dashed', alpha=0.5)
                ax[1].plot(np.real(xuv_rand), color='blue')
                ax[1].plot(np.imag(xuv_rand), color='red')
                plt.pause(0.001)

        else:
            strace = sess.run(image, feed_dict={xuv_input: xuv_rand.reshape(1, -1, 1)})




        # divide the xuv into real and imaginary
        xuv_real = np.real(xuv_rand)
        xuv_imag = np.imag(xuv_rand)

        # append the hdf5 file
        hdf5_file.root.xuv_real.append(xuv_real.reshape(1, -1))
        hdf5_file.root.xuv_imag.append(xuv_imag.reshape(1, -1))
        hdf5_file.root.trace.append(strace.reshape(1, -1))




    hdf5_file.close()


# test open the file
hdf5_file = tables.open_file('attstrace.hdf5', mode='r')

index = 0

xuv = hdf5_file.root.xuv_real[index, :] + 1j * hdf5_file.root.xuv_imag[index, :]


plt.ioff()
plt.cla()
ax[0].pcolormesh(hdf5_file.root.trace[index, :].reshape(len(p_vec), len(tauvec)), cmap='jet')
ax[0].text(0.2, 0.9, 'index: {}'.format(str(index)), transform=ax[0].transAxes, backgroundcolor='white')
ax[1].plot(np.abs(xuv), color='black', linestyle='dashed', alpha=0.5)
ax[1].plot(np.real(xuv), color='blue')
ax[1].plot(np.imag(xuv), color='red')

hdf5_file.close()

plt.show()


