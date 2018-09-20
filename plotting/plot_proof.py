import os
os.chdir('..')
from proof import *
import numpy as np
import matplotlib.pyplot as plt


# plot the IR, xuv, and attosecond
fig = plt.figure(constrained_layout=True, figsize=(7, 7))
gs = fig.add_gridspec(2, 2)

ax = fig.add_subplot(gs[0, 0])
ax.plot(irtmat, irEt, color='orange')
ax.text(0, 0.9, 'IR pulse', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[0, 1])
ax.plot(xuv_int_t, np.real(xuv), color='blue')
ax.text(0, 0.9, 'XUV pulse', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[1, :])
ax.pcolormesh(tauvec_time, p_vec, trace, cmap='jet')
ax.text(0, 0.9, 'Attosecond Streaking Trace', transform=ax.transAxes, backgroundcolor='white')



# plot the filtering
fig = plt.figure(constrained_layout=True, figsize=(7, 7))
gs = fig.add_gridspec(5, 2)

# plot the XUV
ax = fig.add_subplot(gs[0, 0])
ax.plot(xuv_int_t, np.real(xuv), color='blue')
ax.text(0, 0.9, 'XUV', transform=ax.transAxes, backgroundcolor='white')

# plot the trace
ax = fig.add_subplot(gs[0, 1])
ax.pcolormesh(tauvec_time, p_vec, trace, cmap='jet')
ax.text(0, 0.9, 'Attosecond Streaking Trace', transform=ax.transAxes, backgroundcolor='white')

# plot the fourier transform of the trace
ax = fig.add_subplot(gs[1, :])
ax.pcolormesh(tauvec_f_space, p_vec, np.abs(traceft), cmap='jet')
ax.text(0, 0.9, 'Fourier Transform of Trace', transform=ax.transAxes, backgroundcolor='white')

# plot the filter
ax = fig.add_subplot(gs[2, :])
ax.pcolormesh(tauvec_f_space, p_vec, filter2d, cmap='jet')
ax.text(0, 0.8, 'Apply filter', transform=ax.transAxes, backgroundcolor='white')
ax.text(0.58, 0.5, '$+\omega_1$', transform=ax.transAxes, backgroundcolor='white')
ax.text(0.36, 0.5, '$-\omega_1$', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[3, :])
ax.pcolormesh(tauvec_f_space, p_vec, np.abs(trace_filtered), cmap='jet')
ax.text(0, 0.8, 'abs of trace [Frequency Domain]', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[4, :])
ax.pcolormesh(tauvec_time, p_vec, np.abs(trace_filtered_time), cmap='jet')
ax.text(0, 0.8, 'abs of trace [Delay Domain]', transform=ax.transAxes, backgroundcolor='white')


# plot the compensation
fig = plt.figure(constrained_layout=True, figsize=(7, 7))
gs = fig.add_gridspec(3, 2)
ax = fig.add_subplot(gs[0, :])
ax.pcolormesh(tauvec_time, p_vec, np.abs(trace_filtered_time), cmap='jet')
ax.text(0, 0.9, 'abs of trace', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[1, :])
ax.pcolormesh(tauvec_time, p_vec, xuv_spectrum_compensation, cmap='jet')
ax.text(0, 0.9, 'Apply compensation for XUV spectrum', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[2, :])
ax.pcolormesh(tauvec_time, p_vec, np.abs(trace_compensated), cmap='jet')
ax.text(0, 0.9, 'trace with compensation', transform=ax.transAxes, backgroundcolor='white')



# plot only the xuv, attosecond trace, and proof trace
fig = plt.figure(constrained_layout=True, figsize=(7, 7))
gs = fig.add_gridspec(3, 2)
# plot the XUV
ax = fig.add_subplot(gs[0, :])
ax.plot(np.abs(xuv), color='black', linestyle='dashed', alpha=0.5)
ax.plot(np.real(xuv), color='blue')
ax.text(0, 0.9, 'XUV', transform=ax.transAxes, backgroundcolor='white')

# plot the trace
ax = fig.add_subplot(gs[1, :])
ax.pcolormesh(tauvec_time, p_vec, trace, cmap='jet')
ax.text(0, 0.9, 'Attosecond Streaking Trace', transform=ax.transAxes, backgroundcolor='white')

ax = fig.add_subplot(gs[2, :])
ax.pcolormesh(tauvec_time, p_vec, np.abs(trace_filtered_time), cmap='jet')
ax.text(0, 0.8, 'abs of trace [Delay Domain]', transform=ax.transAxes, backgroundcolor='white')






plt.show()


