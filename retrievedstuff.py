import numpy as np
import pickle
import matplotlib.pyplot as plt
import crab_tf2
import tensorflow as tf
xuv_time_domain_func = crab_tf2.xuv_time_domain
import scipy.constants as sc




def plot_Ef_Et(title, Ef):

    Et = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ef)))

    #Ef_start_i = 512 - 100
    #Ef_end_i = 512 + 100
    Ef_start_i = 0
    Ef_end_i = -1

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].plot(f[Ef_start_i:Ef_end_i], np.real(Ef[Ef_start_i:Ef_end_i]), color='blue', label='real')
    ax[0].plot(f[Ef_start_i:Ef_end_i], np.imag(Ef[Ef_start_i:Ef_end_i]), color='red', label='imag')
    ax[0].plot(f[Ef_start_i:Ef_end_i], np.abs(Ef[Ef_start_i:Ef_end_i]), color='black', label='abs')
    ax[0].legend()
    ax[0].set_title(title)
    ax[0].set_xlabel('Hz')
    axtwin = ax[0].twinx()
    axtwin.plot(f[Ef_start_i:Ef_end_i], np.unwrap(np.angle(Ef[Ef_start_i:Ef_end_i])), color='green')
    ax[1].plot(time_as, np.real(Et), color='blue', label='real')
    ax[1].plot(time_as, np.abs(Et), color='black', label='abs')
    ax[1].legend()
    ax[1].set_xlabel('attoseconds')
    plt.savefig(title.replace(' ', '')+'.png')



def plot_Et_Ef(title, Et):

    Ef = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Et)))

    #Ef_start_i = 512 - 100
    #Ef_end_i = 512 + 100
    Ef_start_i = 0
    Ef_end_i = -1

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    ax[0].plot(f[Ef_start_i:Ef_end_i], np.real(Ef[Ef_start_i:Ef_end_i]), color='blue', label='real')
    ax[0].plot(f[Ef_start_i:Ef_end_i], np.imag(Ef[Ef_start_i:Ef_end_i]), color='red', label='imag')
    ax[0].plot(f[Ef_start_i:Ef_end_i], np.abs(Ef[Ef_start_i:Ef_end_i]), color='black', label='abs')
    ax[0].legend()
    ax[0].set_title(title)
    ax[0].set_xlabel('Hz')
    axtwin = ax[0].twinx()
    axtwin.plot(f[Ef_start_i:Ef_end_i], np.unwrap(np.angle(Ef[Ef_start_i:Ef_end_i])), color='green')
    ax[1].plot(time_as, np.real(Et), color='blue', label='real')
    ax[1].plot(time_as, np.abs(Et), color='black', label='abs')
    ax[1].legend()
    ax[1].set_xlabel('attoseconds')
    plt.savefig(title.replace(' ', '')+'.png')








with open('importantstuff.p', 'rb') as file:
    data = pickle.load(file)


print(data.keys())




generated_image = data['generated_image']
input_image = data['input_image']
actual_fields = data['actual_fields']
predicted_fields = data['predicted_fields']


print(predicted_fields.keys())


xuv_f = predicted_fields['xuv_f']

with tf.Session() as sess:
    xuv_predicted_time = sess.run(xuv_time_domain_func, feed_dict={crab_tf2.xuv_cropped_f: predicted_fields['xuv_f']})


#plt.figure(1)
#plt.plot(np.real(xuv_f))


time_as = crab_tf2.xuv.tmat * sc.physical_constants['atomic unit of time'][0] * 1e18




#plt.figure(2)
#plt.plot(time_as, np.real(xuv_predicted_time), color='blue', label='real E')
#plt.plot(time_as, np.abs(xuv_predicted_time), color='black', label='abs E')
#plt.xlabel('attoseconds')
#plt.legend()
#

f = crab_tf2.xuv.fmat /sc.physical_constants['atomic unit of time'][0]

Ef = np.fft.fftshift(np.fft.fft(np.fft.fftshift(xuv_predicted_time)))

#plt.figure(3)
#plt.plot(f, np.real(Ef), color='blue')
#plt.plot(f, np.imag(Ef), color='red')



#plt.figure(4)
#find index max
indexmax = np.argmax(np.abs(Ef))
Ef_centered_0 = np.roll(Ef, (512-indexmax))
Efneg = -1 * Ef_centered_0
Efmirrored = np.flip(Ef_centered_0, axis=0)
Ef_conj = np.real(Ef) + -1j*np.imag(Ef)


# in time domain
#Et = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(Ef)))
#Et_conj = Et



plot_Ef_Et(title='Original E', Ef=Ef)
plot_Ef_Et(title='peak shifted to 0', Ef=Ef_centered_0)
plot_Ef_Et(title='conjugate', Ef=Ef_conj)
#plot_Et_Ef(title='E multiplied by -1', Ef=Efneg)
#plot_Et_Ef(title='E flipped about y axis', Ef=Efmirrored)





plt.show()

##

















plt.show()