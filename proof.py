import numpy as np
import matplotlib.pyplot as plt
import tables
import scipy.constants as sc
import matplotlib.gridspec as gridspec
# from crab_tf import items, xuv_int_t
import pickle


try:
    with open('crab_tf_items.p', 'rb') as file:
        crab_tf_items = pickle.load(file)
    items = crab_tf_items['items']
    xuv_int_t = crab_tf_items['xuv_int_t']

except Exception as e:
    print(e)
    print('run crab_tf.py first to pickle the needed files')
    exit(0)



p_vec = np.linspace(3, 6.5, 200)
tauvec = np.arange(-22000, 22000, 250)

# construct frequency axis
# PULLING THESE VALS FROM crab_tf.py
tmax = 60e-15
N = 2**16
dt = tmax / N
dt = dt / sc.physical_constants['atomic unit of time'][0]
tauvec_time = tauvec * dt

dtau = tauvec_time[1] - tauvec_time[0]
Ntau = len(tauvec_time)

df_tau = 1 / (Ntau * dtau) # frequency in au
tauvec_f_space = df_tau * np.arange(-Ntau/2, Ntau/2, 1)



# open the file and retrieve the trace
index = 0
hdf5_file = tables.open_file('attstrace.hdf5', mode='r')
xuv = hdf5_file.root.xuv_real[index, :] + 1j * hdf5_file.root.xuv_imag[index, :]
trace = hdf5_file.root.trace[index, :].reshape(len(p_vec), len(tauvec))
hdf5_file.close()


# fourier transform the trace
print(np.shape(trace))
traceft = np.fft.fftshift(np.fft.fft(np.fft.fftshift(trace, axes=1), axis=1), axes=1)



# convert IR driving freq to attomic
lam0 = 1.7 * 1e-6
f0_ir = sc.c / lam0
f0_ir = f0_ir * sc.physical_constants['atomic unit of time'][0]

# tauvec_f_space
print(f0_ir)

# construct filter
f_center = f0_ir
width = 0.001

#find index of positive frequency
filter_type = 'rect'

if filter_type == 'rect':
    width = 2
    filter = np.zeros_like(tauvec_f_space)
    f_pos_index = np.argmin(np.abs(tauvec_f_space - f0_ir))
    f_neg_index = np.argmin(np.abs(tauvec_f_space + f0_ir))
    filter[f_pos_index-width:f_pos_index+width] = 1
    filter[f_neg_index-width:f_neg_index+width] = 1

elif filter_type =='gaussian':

    filter = np.exp(-(tauvec_f_space - f_center)**2 / width**2)
    filter += np.exp(-(tauvec_f_space + f_center)**2 / width**2)


filter2d = filter.reshape(1, -1) * np.real(np.ones_like(traceft))
trace_filtered = filter2d * traceft

trace_filtered_time = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(trace_filtered, axes=1), axis=1), axes=1)


# spectrum of xuv for normalization of image
K = (0.5 * p_vec**2).reshape(-1, 1)
## importing these from crab_tf .. should import everyhing now.
e_fft = np.exp(-1j * (K + items['Ip']) * xuv_int_t.reshape(1, -1))
product = xuv.reshape(1, -1) * e_fft
integral = np.abs(dt * np.sum(product, axis=1))**2

# need to figure out this scaling later
scaling = 1
compensation = 1 + scaling * (np.max(integral) - integral)

xuv_spectrum_compensation = np.ones_like(trace) * compensation.reshape(-1, 1)

trace_compensated = np.abs(trace_filtered_time) * xuv_spectrum_compensation





# plot the trace before and after filtering
fig = plt.figure(constrained_layout=True, figsize=(10, 7))
gs = fig.add_gridspec(5, 3)

ax = fig.add_subplot(gs[0, 0])
ax.plot(np.abs(xuv), color='black', linestyle='dashed', alpha=0.5)
ax.plot(np.real(xuv), color='blue')
ax.plot(np.imag(xuv), color='red')

ax = fig.add_subplot(gs[0, 1])
ax.pcolormesh(trace, cmap='jet')
ax.text(0.2, 0.9, 'index: {}'.format(str(index)), transform=ax.transAxes, backgroundcolor='white')


ax = fig.add_subplot(gs[1, 0])
ax.pcolormesh(tauvec_f_space, p_vec, np.real(traceft), cmap='jet')
ax.text(0, 0.8, 'real part of trace', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[1, 1])
ax.pcolormesh(tauvec_f_space, p_vec, np.imag(traceft), cmap='jet')
ax.text(0, 0.8, 'imag part of trace', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[1, 2])
ax.pcolormesh(tauvec_f_space, p_vec, np.abs(traceft), cmap='jet')
ax.text(0, 0.8, 'abs of trace', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[2, 1])
ax.pcolormesh(tauvec_f_space, p_vec, filter2d, cmap='jet')
ax.text(0, 0.8, 'filter', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[2, 0])
plt.plot(tauvec_f_space, filter, color='blue')
ax.text(0, 0.8, 'filter', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[3, 0])
ax.pcolormesh(tauvec_f_space, p_vec, np.real(trace_filtered), cmap='jet')
ax.text(0, 0.8, 'real part of trace', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[3, 1])
ax.pcolormesh(tauvec_f_space, p_vec, np.imag(trace_filtered), cmap='jet')
ax.text(0, 0.8, 'imag part of trace', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[3, 2])
ax.pcolormesh(tauvec_f_space, p_vec, np.abs(trace_filtered), cmap='jet')
ax.text(0, 0.8, 'abs of trace', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[4, 0])
ax.pcolormesh(tauvec_f_space, p_vec, np.real(trace_filtered_time), cmap='jet')
ax.text(0, 0.8, 'real part of trace', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[4, 1])
ax.pcolormesh(tauvec_f_space, p_vec, np.imag(trace_filtered_time), cmap='jet')
ax.text(0, 0.8, 'imag part of trace', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[4, 2])
ax.pcolormesh(tauvec_f_space, p_vec, np.abs(trace_filtered_time), cmap='jet')
ax.text(0, 0.8, 'abs of trace', transform=ax.transAxes, backgroundcolor='white')



# plot the trace before and after copmensation
tau_cross_section = 94
fig = plt.figure(constrained_layout=True, figsize=(10, 7))
gs = fig.add_gridspec(5, 2)

ax = fig.add_subplot(gs[0, :])
ax.pcolormesh(tauvec_f_space, p_vec, np.abs(trace_filtered_time), cmap='jet')
ax.text(0, 0.8, 'abs of trace', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[1, 0])
vals = np.abs(trace_filtered_time)
p_slice = np.array(vals[:, tau_cross_section])
vals[:, tau_cross_section] = np.max(vals)
ax.pcolormesh(tauvec_f_space, p_vec, vals, cmap='jet')
ax.text(0, 0.8, 'trace with compensation', transform=ax.transAxes, backgroundcolor='white')
ax = fig.add_subplot(gs[1, 1])
ax.plot(p_vec, p_slice)


ax = fig.add_subplot(gs[2, 0])
ax.pcolormesh(tauvec_f_space, p_vec, xuv_spectrum_compensation, cmap='jet')
ax.text(0, 0.8, 'xuv spectral compensation', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[2, 1])
ax.plot(p_vec, compensation)
ax.text(0, -0.1, 'xuv spectral compensation scaling:{}'.format(scaling), transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[3, :])
ax.pcolormesh(tauvec_f_space, p_vec, np.abs(trace_compensated), cmap='jet')
ax.text(0, 0.8, 'trace with compensation', transform=ax.transAxes, backgroundcolor='white')

# plot a cross section of the values
ax = fig.add_subplot(gs[4, 0])
vals = np.abs(trace_compensated)
p_slice = np.array(vals[:, tau_cross_section])
vals[:, tau_cross_section] = np.max(vals)
ax.pcolormesh(tauvec_f_space, p_vec, vals, cmap='jet')
ax.text(0, 0.8, 'trace with compensation', transform=ax.transAxes, backgroundcolor='white')
ax = fig.add_subplot(gs[4, 1])
ax.plot(p_vec, p_slice)

plt.show()

