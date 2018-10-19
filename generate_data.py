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

        # make sure the information can be saved

        #construct frequency axis
        dt = self.tmat_cropped[1] - self.tmat_cropped[0]
        N_cropped = len(self.tmat_cropped)
        df_cropped = 1 / (N_cropped * dt)
        f_cropped = df_cropped * np.arange(-N_cropped/2, N_cropped/2, 1)

        # crop the spectrum
        start_index = 260
        width = 64


        self.f_cropped_cropped = f_cropped[start_index:start_index+width]
        self.Ef_cropped_cropped = self.Et_cropped_f_phase[start_index:start_index+width]


        self.E_w64 = self.Ef_cropped_cropped

        # set the phase angle to 0 at center
        w0_index = 32
        angle_at_w0 = np.angle(self.E_w64[w0_index])
        self.E_w64 = self.E_w64 * np.exp(-1j * angle_at_w0)

        # construct 512 timestep Et from 64 timestep Ef
        self.E_t512 = Ew_64_to_Et_512(self.Ef_cropped_cropped, f_cropped,
                             start_index, width)




def generate_samples(n_samples, filename):


    # create hdf5 file
    print('creating file: '+filename)
    hdf5_file = tables.open_file(filename, mode='w')
    hdf5_file.create_earray(hdf5_file.root,
                                           'trace', tables.Float16Atom(), shape=(0, len(p_vec) * len(tauvec)))
    hdf5_file.create_earray(hdf5_file.root,
                                       'xuv_real', tables.Float16Atom(), shape=(0, len(xuv_test)))
    hdf5_file.create_earray(hdf5_file.root,
                                       'xuv_imag', tables.Float16Atom(), shape=(0, len(xuv_test)))
    hdf5_file.create_earray(hdf5_file.root,
                            'xuv_frequency_domain', tables.ComplexAtom(itemsize=16),
                            shape=(0, 64))

    hdf5_file.close()


    # open the hdf5 file
    hdf5_file = tables.open_file(filename, mode='a')

    for i in range(n_samples):

        # generate a random xuv pulse
        xuv_object = XUV_Field_rand_phase(phase_amplitude=5, phase_nodes=120, plot=False)

        xuv_rand = xuv_object.E_t512
        xuv_rand_f = xuv_object.E_w64


        if i % 500 == 0:
            print('generating sample {} of {}'.format(i + 1, n_samples))
            # generate the FROG trace
            time1 = time.time()
            strace = sess.run(image, feed_dict={xuv_input: xuv_rand.reshape(1, -1, 1)})
            time2 = time.time()
            duration = time2 - time1
            print('duration: {} s'.format(round(duration, 4)))

            if plotting:

                ax[0].clear()
                ax[0].pcolormesh(strace, cmap='jet')

                ax[1].clear()
                ax[1].plot(np.abs(xuv_rand), color='black', linestyle='dashed', alpha=0.5)
                ax[1].plot(np.real(xuv_rand), color='blue')
                ax[1].plot(np.imag(xuv_rand), color='red')

                ax[2].clear()
                ax[2].plot(np.real(xuv_rand_f), color='blue')
                ax[2].plot(np.imag(xuv_rand_f), color='red')

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
        hdf5_file.root.xuv_frequency_domain.append(xuv_rand_f.reshape(1, -1))


    hdf5_file.close()



def Ew_64_to_Et_512(E_f_64, f_512, start_index, width):

    E_f_512 = np.zeros_like(f_512, dtype=complex)
    E_f_512[start_index:start_index+width] = E_f_64
    # convert to time domain
    E_t = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(E_f_512)))
    return E_t


xuv_test = XUV_Field(N=N, tmax=tmax, gdd=500, tod=0).Et[lower:upper]


fig, ax = plt.subplots(3, 1)

plt.ion()
plotting = True
init = tf.global_variables_initializer()
with tf.Session() as sess:

    n_train_samples = 20000
    n_test_samples = 500

    init.run()

    generate_samples(n_samples=n_train_samples, filename='attstrace_train.hdf5')
    generate_samples(n_samples=n_test_samples, filename='attstrace_test.hdf5')


# test open the file
index = 20
hdf5_file = tables.open_file('attstrace_train.hdf5', mode='r')
# hdf5_file = tables.open_file('attstrace_test.hdf5', mode='r')

xuv = hdf5_file.root.xuv_real[index, :] + 1j * hdf5_file.root.xuv_imag[index, :]
xuv_f = hdf5_file.root.xuv_frequency_domain[index, :]
plt.ioff()

ax[0].clear()
ax[0].pcolormesh(hdf5_file.root.trace[index, :].reshape(len(p_vec), len(tauvec)), cmap='jet')
ax[0].text(0.2, 0.9, 'index: {}'.format(str(index)), transform=ax[0].transAxes, backgroundcolor='white')

ax[1].clear()
ax[1].plot(np.abs(xuv), color='black', linestyle='dashed', alpha=0.5)
ax[1].plot(np.real(xuv), color='blue')
ax[1].plot(np.imag(xuv), color='red')

ax[2].clear()
ax[2].plot(np.real(xuv_f), color='blue')
ax[2].plot(np.imag(xuv_f), color='red')

hdf5_file.close()

plt.show()


