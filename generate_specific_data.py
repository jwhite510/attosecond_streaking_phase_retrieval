from crab_tf import *
import tensorflow as tf
from scipy.interpolate import interp1d
import tables
import time


class XUV_Field_specific_phase(XUV_Field):

    def __init__(self):

        XUV_Field.__init__(self, N=N, tmax=tmax, gdd=0, tod=0)

        middle_index = int(len(self.Et) / 2)
        lower = middle_index-int(self.span/2)
        upper = middle_index+int(self.span/2)

        # define the integration timespan of the field
        self.Et_cropped = self.Et[lower:upper]

        # fft the pulse
        self.Et_cropped_f = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.Et_cropped)))

        self.cropped_df = 1/(self.dt * len(self.Et_cropped))
        self.cropped_t = self.tmat[lower:upper]
        self.cropped_f = self.cropped_df * np.arange(-len(self.cropped_t)/2, len(self.cropped_t)/2, 1)

    def apply_phase(self, spectral_phase):
        E_w_phase = self.Et_cropped_f * np.exp(1j * spectral_phase)
        E_t_phase = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(E_w_phase)))
        return E_t_phase




xuv_test = XUV_Field(N=N, tmax=tmax, gdd=500, tod=0).Et[lower:upper]


test_open = False


filename = 'attstrac_specific.hdf5'
# create hdf5 file
hdf5_file = tables.open_file(filename, mode='w')
frog_image_f = hdf5_file.create_earray(hdf5_file.root,
                                       'trace', tables.Float16Atom(), shape=(0, len(p_vec) * len(tauvec)))
E_real_f = hdf5_file.create_earray(hdf5_file.root,
                                   'xuv_real', tables.Float16Atom(), shape=(0, len(xuv_test)))
E_imag_f = hdf5_file.create_earray(hdf5_file.root,
                                   'xuv_imag', tables.Float16Atom(), shape=(0, len(xuv_test)))
hdf5_file.close()



# generate data
xuv_specific = XUV_Field_specific_phase()
# generate phase
fmat = xuv_specific.cropped_f


xuv_specific_phase_pulses = []

# pulse 0
gdd = 0
phase = gdd * (fmat - xuv_specific.f0)**2
E_t_phase = xuv_specific.apply_phase(spectral_phase=phase)
xuv_specific_phase_pulses.append(E_t_phase)

# pulse 1
gdd = -30
phase = gdd * (fmat - xuv_specific.f0)**2
E_t_phase = xuv_specific.apply_phase(spectral_phase=phase)
xuv_specific_phase_pulses.append(E_t_phase)

# pulse 2
gdd = 30
phase = gdd * (fmat - xuv_specific.f0)**2
E_t_phase = xuv_specific.apply_phase(spectral_phase=phase)
xuv_specific_phase_pulses.append(E_t_phase)

# pulse 3
gdd = 0
tod = 20
phase = gdd * (fmat - xuv_specific.f0)**2 + tod * (fmat - xuv_specific.f0)**3
E_t_phase = xuv_specific.apply_phase(spectral_phase=phase)
xuv_specific_phase_pulses.append(E_t_phase)

# pulse 4
gdd = 0
tod = -20
phase = gdd * (fmat - xuv_specific.f0)**2 + tod * (fmat - xuv_specific.f0)**3
E_t_phase = xuv_specific.apply_phase(spectral_phase=phase)
xuv_specific_phase_pulses.append(E_t_phase)


"""
test for ambiguities
"""
# pulse 5
gdd = 0
tod = -20
phase = gdd * (fmat - xuv_specific.f0)**2 + tod * (fmat - xuv_specific.f0)**3
E_t_phase = xuv_specific.apply_phase(spectral_phase=phase)
xuv_specific_phase_pulses.append(E_t_phase)


# add constant phase shift 6
phase = 0.25 * np.pi
E_t_phase_ps = E_t_phase * np.exp(1j * phase)
xuv_specific_phase_pulses.append(E_t_phase_ps)


# add constant phase shift 7
phase = 0.5 * np.pi
E_t_phase_ps = E_t_phase * np.exp(1j * phase)
xuv_specific_phase_pulses.append(E_t_phase_ps)


# add constant phase shift 8
phase = 0.75 * np.pi
E_t_phase_ps = E_t_phase * np.exp(1j * phase)
xuv_specific_phase_pulses.append(E_t_phase_ps)



# add constant phase shift 9
phase = 1 * np.pi
E_t_phase_ps = E_t_phase * np.exp(1j * phase)
xuv_specific_phase_pulses.append(E_t_phase_ps)



# plt.figure(5)
# plt.plot(np.real(E_t_phase))
# plt.show()
# exit(0)




# """
# pulse 5 - 14 translated in time
# """
# # translate in time
# gdd = 0
# tod = -20
# for time_translate in np.linspace(-5, 5, 10):
#
#     phase = 2 * np.pi * fmat * time_translate
#     phase += gdd * (fmat - xuv_specific.f0)**2 + tod * (fmat - xuv_specific.f0)**3
#     E_t_phase = xuv_specific.apply_phase(spectral_phase=phase)
#     xuv_specific_phase_pulses.append(E_t_phase)












fig, ax = plt.subplots(2, 1)
plt.ion()
plotting = True
init = tf.global_variables_initializer()
with tf.Session() as sess:

    init.run()

    # open the hdf5 file
    hdf5_file = tables.open_file(filename, mode='a')

    for xuv_specific_pulse, i in zip(xuv_specific_phase_pulses, range(len(xuv_specific_phase_pulses))):


        print('generating sample {} of {}'.format(i + 1, len(xuv_specific_phase_pulses)))
        # generate the FROG trace
        time1 = time.time()
        strace = sess.run(image, feed_dict={xuv_input: xuv_specific_pulse.reshape(1, -1, 1)})
        time2 = time.time()
        duration = time2 - time1
        print('duration: {} s'.format(round(duration, 4)))

        if plotting:
            plt.cla()
            ax[0].pcolormesh(strace, cmap='jet')
            ax[1].plot(np.abs(xuv_specific_pulse), color='black', linestyle='dashed', alpha=0.5)
            ax[1].plot(np.real(xuv_specific_pulse), color='blue')
            ax[1].plot(np.imag(xuv_specific_pulse), color='red')
            plt.pause(0.001)


        # divide the xuv into real and imaginary
        xuv_real = np.real(xuv_specific_pulse)
        xuv_imag = np.imag(xuv_specific_pulse)

        # append the hdf5 file
        hdf5_file.root.xuv_real.append(xuv_real.reshape(1, -1))
        hdf5_file.root.xuv_imag.append(xuv_imag.reshape(1, -1))
        hdf5_file.root.trace.append(strace.reshape(1, -1))




    hdf5_file.close()



if test_open:
    # test open the file
    hdf5_file = tables.open_file(filename, mode='r')

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


