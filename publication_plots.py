import crab_tf2
import pickle
xuv_time_domain_func = crab_tf2.xuv_time_domain
import shutil
import network2
import tensorflow as tf
import glob
import tables
import numpy as np
import  matplotlib.pyplot as plt
import os
import csv
from scipy import interpolate
import scipy.constants as sc
from matplotlib.ticker import MaxNLocator





def plot_single_trace_xuv_only_measured(predicted_xuv, actual_xuv, trace, reconstructed_trace, name,
                      plot_streaking_mse, measured_spec):

    letters = ['a)', 'b)', 'c)', 'd)']
    letter_index = 0

    # initialize the plot for single trace
    fig = plt.figure(figsize=(5,9))
    borderspace_lr = 0.15
    borderspace_tb = 0.08

    plt.subplots_adjust(left=0+borderspace_lr, right=1-borderspace_lr,
                        top=1-borderspace_tb, bottom=0+borderspace_tb,
                        hspace=0.45, wspace=0.4)
    gs = fig.add_gridspec(3, 1)

    # generate time for plotting
    si_time = crab_tf2.tau_values * sc.physical_constants['atomic unit of time'][0]

    # predictions = sess.run(y_pred, feed_dict={x: x_in, y_true: y_in})
    # mse = sess.run(loss, feed_dict={x: x_in[index].reshape(1, -1), y_true: y_in[index].reshape(1, -1)})


    # plot input trace
    axis = fig.add_subplot(gs[0, :])
    axis.pcolormesh(si_time * 1e15, crab_tf2.eV_values, trace, cmap='jet')
    axis.set_xlabel('Time Delay [fs]')
    axis.set_ylabel('Energy [eV]')
    # axis.text(0.5, 1.06, "Input Streaking Trace", transform=axis.transAxes, backgroundcolor='white', weight='bold',
    #           horizontalalignment='center')
    # axis.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    # subplot just for the letter for input trace
    axis.text(0, 1.06, letters[letter_index]+" Input Streaking Trace", transform=axis.transAxes, backgroundcolor='white', weight='bold')
    letter_index +=1





    # plotting generated trace
    axis = fig.add_subplot(gs[1, :])
    axis.pcolormesh(si_time * 1e15, crab_tf2.eV_values,
                    reconstructed_trace, cmap='jet')
    axis.set_xlabel('Time Delay [fs]')
    axis.set_ylabel('Energy [eV]')
    # axis.text(0.5, 1.06, "Reconstructed Streaking Trace", transform=axis.transAxes, backgroundcolor='white',
    #           weight='bold', horizontalalignment='center')
    # subplot just for the letter for input trace
    axis.text(0, 1.06, letters[letter_index]+" Reconstructed Streaking Trace", transform=axis.transAxes, backgroundcolor='white', weight='bold')
    letter_index += 1
    # plot trace MSE
    if plot_streaking_mse:
        trace_reconstructed_reshaped = reconstructed_trace.reshape(-1)
        trace_actual_reshape = trace.reshape(-1)
        trace_mse = (1/len(trace_actual_reshape)) * np.sum((trace_reconstructed_reshaped - trace_actual_reshape)**2)
        print('trace_mse: ', trace_mse)
        axis.text(0.012, 0.96, "MSE: " + str(round(trace_mse, 5)), transform=axis.transAxes, backgroundcolor='white',
                    bbox=dict(facecolor='white', edgecolor='black', pad=3.0))



    # plot the predicted xuv spectral phase
    # determine xuv phase crop
    crop_phase_right = global_xuv_phase_crop_right
    crop_phase_left = global_xuv_phase_crop_left
    axis = fig.add_subplot(gs[2, :])
    axis.cla()
    axis.plot(electronvolts, np.abs(predicted_xuv) ** 2, color="black", label='Predicted Spectrum')
    axis.plot(electronvolts, measured_spec, color="red", label='Measured Spectrum')
    axtwin = axis.twinx()
    # plot the phase
    phase = np.unwrap(np.angle(predicted_xuv[crop_phase_left:-crop_phase_right]))
    phase = phase - np.min(phase)
    axtwin.plot(electronvolts[crop_phase_left:-crop_phase_right], phase, color="green", linewidth=3)
    # set ticks
    tickmax = int(np.max(phase))
    tickmin = int(np.min(phase))
    # ticks = np.arange(0, tickmax + 1, 5)
    #    axtwin.set_yticks(ticks)
    axtwin.set_ylabel(r"$\phi_{XUV}$[rad]")
    axtwin.yaxis.label.set_color('green')
    axtwin.tick_params(axis='y', colors='green')
    axtwin.yaxis.set_major_locator(MaxNLocator(integer=True))
    # plot the error
    axis.text(0, 1.05, letters[letter_index] + " Predicted XUV", transform=axis.transAxes, backgroundcolor='white',
              weight='bold')
    letter_index += 1
    axis.set_xlabel('Photon Energy [eV]')
    axis.set_ylabel('Intensity [arbitrary units]')
    #axis.legend(loc=1).set_zorder(0)
    # axis.legend(loc=4)
    axis.legend(bbox_to_anchor=(0.7, 0.65), loc='center')

    plt.savefig('./'+name)







def get_run(filename):
    #with open('./12_28_18_pictures/' + test_name) as file:
    with open(filename) as file:
        reader = csv.reader(file)
        step, mse = [], []
        for row in np.array(list(reader))[1:]:
            step.append(float(row[1]))
            mse.append(float(row[2]))
        mse = np.array(mse)
        step = np.array(step)

    return step, mse





def plot_mse_steps(filenames, name):

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    colors = ["r", "#EA530E", "#401F3E", "#FF5964", "#083D77", "#151918", "#ED241A"]
    colors1 = []
    for color in colors:
        colors1.append(color)
        colors1.append(color)

    for i, filename, color in zip(range(len(filenames)), filenames, colors1):

        step, mse = get_run(filename)

        if 'train' in filename:
            # plot with label
            run_name = 'Train Data MSE'
            ax.plot(step, mse, label=run_name, color=color)
            ax.set_yscale('log')
        else:
            ax.plot(step, mse, linestyle=':', color=color, linewidth=3)
            ax.set_yscale('log')

        #ax.set_ylim(0, 0.004)
        #ax.set_xlim(-20, 1100)
        ax.set_xlabel('Step')
        ax.set_ylabel('MSE')
        ax.legend()
        plt.savefig('./'+name)








def calc_mse(vec1, vec2):

    real1 = np.real(vec1)
    imag1 = np.imag(vec1)
    vec1_concat = np.append(real1, imag1)

    real2 = np.real(vec2)
    imag2 = np.imag(vec2)
    vec2_concat = np.append(real2, imag2)

    print('lengths')
    print(len(real1))
    print(len(vec1_concat))

    mse = (1/len(vec1_concat)) * np.sum((vec1_concat - vec2_concat)**2)

    return mse






def plot_single_trace_xuv_only_square(predicted_xuv, actual_xuv, trace, reconstructed_trace, name,
                      plot_streaking_mse):

    letters = ['a)', 'b)', 'c)', 'd)']
    letter_index = 0

    # initialize the plot for single trace
    fig = plt.figure(figsize=(8,8))
    borderspace_lr = 0.1
    borderspace_tb = 0.08

    plt.subplots_adjust(left=0+borderspace_lr, right=1-borderspace_lr,
                        top=1-borderspace_tb, bottom=0+borderspace_tb,
                        hspace=0.3, wspace=0.4)
    gs = fig.add_gridspec(2, 2)

    # generate time for plotting
    si_time = crab_tf2.tau_values * sc.physical_constants['atomic unit of time'][0]

    # predictions = sess.run(y_pred, feed_dict={x: x_in, y_true: y_in})
    # mse = sess.run(loss, feed_dict={x: x_in[index].reshape(1, -1), y_true: y_in[index].reshape(1, -1)})

    # plot input trace
    axis = fig.add_subplot(gs[0, 0])
    axis.pcolormesh(si_time * 1e15, crab_tf2.eV_values, trace, cmap='jet')
    axis.set_xlabel('Time Delay [fs]')
    axis.set_ylabel('Energy [eV]')
    # axis.text(0.5, 1.06, "Input Streaking Trace", transform=axis.transAxes, backgroundcolor='white', weight='bold',
    #           horizontalalignment='center')
    # axis.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    # subplot just for the letter for input trace
    axis.text(0, 1.06, letters[letter_index]+" Input Streaking Trace", transform=axis.transAxes, backgroundcolor='white', weight='bold')
    letter_index += 1



    if actual_xuv is not None:
        # plot the actual xuv spectral phase
        # determine xuv phase crop
        crop_phase_right = global_xuv_phase_crop_right
        crop_phase_left = global_xuv_phase_crop_left
        axis = fig.add_subplot(gs[0, 1])
        axis.cla()
        axis.plot(electronvolts, np.abs(actual_xuv) ** 2, color="black")
        axtwin = axis.twinx()
        # plot the phase
        phase = np.unwrap(np.angle(actual_xuv[crop_phase_left:-crop_phase_right]))
        # set ticks
        tickmax = int(np.max(phase))
        tickmin = int(np.min(phase))
        # ticks = np.arange(0, tickmax + 1, 1)
        # axtwin.set_yticks(ticks)
        phase = phase - np.min(phase)
        axtwin.plot(electronvolts[crop_phase_left:-crop_phase_right], phase, color="green", linewidth=3)
        axtwin.set_ylabel(r"$\phi_{XUV}$[rad]")
        axtwin.yaxis.label.set_color('green')
        axtwin.tick_params(axis='y', colors='green')
        axtwin.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.text(0, 1.05, letters[letter_index]+" Actual XUV", transform=axis.transAxes, backgroundcolor='white', weight='bold')
        letter_index+=1
        axis.set_xlabel('Photon Energy [eV]')
        # axis.set_ylabel('$|E_{XUV}(eV)|$')
        axis.set_ylabel('Intensity [arbitrary units]')


    # plotting generated trace
    axis = fig.add_subplot(gs[1, 0])
    axis.pcolormesh(si_time * 1e15, crab_tf2.eV_values,
                    reconstructed_trace, cmap='jet')
    axis.set_xlabel('Time Delay [fs]')
    axis.set_ylabel('Energy [eV]')
    # axis.text(0.5, 1.06, "Reconstructed Streaking Trace", transform=axis.transAxes, backgroundcolor='white',
    #           weight='bold', horizontalalignment='center')
    # subplot just for the letter for input trace
    axis.text(0, 1.05, letters[letter_index]+" Reconstructed Streaking Trace", transform=axis.transAxes, backgroundcolor='white', weight='bold')
    letter_index += 1
    # plot trace MSE
    if plot_streaking_mse:
        trace_reconstructed_reshaped = reconstructed_trace.reshape(-1)
        trace_actual_reshape = trace.reshape(-1)
        trace_mse = (1 / len(trace_actual_reshape)) * np.sum(
            (trace_reconstructed_reshaped - trace_actual_reshape) ** 2)
        print('trace_mse: ', trace_mse)
        axis.text(0.012, 0.96, "MSE: " + str(round(trace_mse, 5)), transform=axis.transAxes,
                  backgroundcolor='white',
                  bbox=dict(facecolor='white', edgecolor='black', pad=3.0))


    # plot the predicted xuv spectral phase
    # determine xuv phase crop
    crop_phase_right = global_xuv_phase_crop_right
    crop_phase_left = global_xuv_phase_crop_left
    axis = fig.add_subplot(gs[1, 1])
    axis.cla()
    axis.plot(electronvolts, np.abs(predicted_xuv)**2, color="black")
    axtwin = axis.twinx()
    # plot the phase
    phase = np.unwrap(np.angle(predicted_xuv[crop_phase_left:-crop_phase_right]))
    phase = phase - np.min(phase)
    axtwin.plot(electronvolts[crop_phase_left:-crop_phase_right],phase, color="green", linewidth=3)
    # set ticks
    tickmax = int(np.max(phase))
    tickmin = int(np.min(phase))
    #ticks = np.arange(0, tickmax + 1, 5)
#    axtwin.set_yticks(ticks)

    axtwin.set_ylabel(r"$\phi_{XUV}$[rad]")
    axtwin.yaxis.label.set_color('green')
    axtwin.tick_params(axis='y', colors='green')
    axtwin.yaxis.set_major_locator(MaxNLocator(integer=True))
    # plot the error
    #mse between xuv traces
    if actual_xuv is not None:
        xuv_mse = calc_mse(predicted_xuv, actual_xuv)
        axtwin.text(0.012, 0.96, "MSE: " + str(round(xuv_mse, 5)), transform=axtwin.transAxes, backgroundcolor='white',
                    bbox=dict(facecolor='white', edgecolor='black', pad=3.0))
    axis.text(0, 1.05, letters[letter_index]+" Predicted XUV", transform=axis.transAxes, backgroundcolor='white', weight='bold')
    letter_index += 1
    axis.set_xlabel('Photon Energy [eV]')
    axis.set_ylabel('Intensity [arbitrary units]')




    plt.savefig('./'+name)




def plot_single_trace_xuv_only(predicted_xuv, actual_xuv, trace, reconstructed_trace, name,
                      plot_streaking_mse):

    letters = ['a)', 'b)', 'c)', 'd)']
    letter_index = 0

    # initialize the plot for single trace
    fig = plt.figure(figsize=(10,10))
    borderspace_lr = 0.1
    borderspace_tb = 0.05

    plt.subplots_adjust(left=0+borderspace_lr, right=1-borderspace_lr,
                        top=1-borderspace_tb, bottom=0+borderspace_tb,
                        hspace=0.4, wspace=0.4)
    gs = fig.add_gridspec(4, 4)

    # generate time for plotting
    si_time = crab_tf2.tau_values * sc.physical_constants['atomic unit of time'][0]

    # predictions = sess.run(y_pred, feed_dict={x: x_in, y_true: y_in})
    # mse = sess.run(loss, feed_dict={x: x_in[index].reshape(1, -1), y_true: y_in[index].reshape(1, -1)})


    if actual_xuv is not None:
        # plot the actual xuv spectral phase
        # determine xuv phase crop
        crop_phase_right = global_xuv_phase_crop_right
        crop_phase_left = global_xuv_phase_crop_left
        axis = fig.add_subplot(gs[0, 1:3])
        axis.cla()
        axis.plot(electronvolts, np.abs(actual_xuv) ** 2, color="black")
        axtwin = axis.twinx()
        # plot the phase
        phase = np.unwrap(np.angle(actual_xuv[crop_phase_left:-crop_phase_right]))
        # set ticks
        tickmax = int(np.max(phase))
        tickmin = int(np.min(phase))
        # ticks = np.arange(0, tickmax + 1, 1)
        # axtwin.set_yticks(ticks)
        phase = phase - np.min(phase)
        axtwin.plot(electronvolts[crop_phase_left:-crop_phase_right], phase, color="green", linewidth=3)
        axtwin.set_ylabel(r"$\phi_{XUV}$[rad]")
        axtwin.yaxis.label.set_color('green')
        axtwin.tick_params(axis='y', colors='green')
        axtwin.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.text(0, 1.05, letters[letter_index]+" Actual XUV", transform=axis.transAxes, backgroundcolor='white', weight='bold')
        letter_index+=1
        axis.set_xlabel('Photon Energy [eV]')
        # axis.set_ylabel('$|E_{XUV}(eV)|$')
        axis.set_ylabel('Intensity [arbitrary units]')


    # plot input trace
    axis = fig.add_subplot(gs[1, :])
    axis.pcolormesh(si_time * 1e15, crab_tf2.eV_values, trace, cmap='jet')
    axis.set_xlabel('Time Delay [fs]')
    axis.set_ylabel('Energy [eV]')
    axis.text(0.5, 1.06, "Input Streaking Trace", transform=axis.transAxes, backgroundcolor='white', weight='bold',
              horizontalalignment='center')
    # axis.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    # subplot just for the letter for input trace
    axis.text(0, 1.06, letters[letter_index], transform=axis.transAxes, backgroundcolor='white', weight='bold')
    letter_index +=1



    # plot the predicted xuv spectral phase
    # determine xuv phase crop
    crop_phase_right = global_xuv_phase_crop_right
    crop_phase_left = global_xuv_phase_crop_left
    axis = fig.add_subplot(gs[2,1:3])
    axis.cla()
    axis.plot(electronvolts, np.abs(predicted_xuv)**2, color="black")
    axtwin = axis.twinx()
    # plot the phase
    phase = np.unwrap(np.angle(predicted_xuv[crop_phase_left:-crop_phase_right]))
    phase = phase - np.min(phase)
    axtwin.plot(electronvolts[crop_phase_left:-crop_phase_right],phase, color="green", linewidth=3)
    # set ticks
    tickmax = int(np.max(phase))
    tickmin = int(np.min(phase))
    #ticks = np.arange(0, tickmax + 1, 5)
#    axtwin.set_yticks(ticks)

    axtwin.set_ylabel(r"$\phi_{XUV}$[rad]")
    axtwin.yaxis.label.set_color('green')
    axtwin.tick_params(axis='y', colors='green')
    axtwin.yaxis.set_major_locator(MaxNLocator(integer=True))
    # plot the error
    #mse between xuv traces
    if actual_xuv is not None:
        xuv_mse = calc_mse(predicted_xuv, actual_xuv)
        axtwin.text(0.03, 0.90, "MSE: " + str(round(xuv_mse, 5)), transform=axtwin.transAxes, backgroundcolor='white',
                    bbox=dict(facecolor='white', edgecolor='black', pad=3.0))
    axis.text(0, 1.05, letters[letter_index]+" Predicted XUV", transform=axis.transAxes, backgroundcolor='white', weight='bold')
    letter_index += 1
    axis.set_xlabel('Photon Energy [eV]')
    axis.set_ylabel('Intensity [arbitrary units]')


    # plotting generated trace
    axis = fig.add_subplot(gs[3, :])
    axis.pcolormesh(si_time * 1e15, crab_tf2.eV_values,
                    reconstructed_trace, cmap='jet')
    axis.set_xlabel('Time Delay [fs]')
    axis.set_ylabel('Energy [eV]')
    axis.text(0.5, 1.06, "Reconstructed Streaking Trace", transform=axis.transAxes, backgroundcolor='white',
              weight='bold', horizontalalignment='center')
    # subplot just for the letter for input trace
    axis.text(0, 1.05, letters[letter_index], transform=axis.transAxes, backgroundcolor='white', weight='bold')
    letter_index += 1


    # plot trace MSE
    if plot_streaking_mse:
        trace_reconstructed_reshaped = reconstructed_trace.reshape(-1)
        trace_actual_reshape = trace.reshape(-1)
        trace_mse = (1/len(trace_actual_reshape)) * np.sum((trace_reconstructed_reshaped - trace_actual_reshape)**2)
        print('trace_mse: ', trace_mse)
        axis.text(0.012, 0.90, "MSE: " + str(round(trace_mse, 5)), transform=axis.transAxes, backgroundcolor='white',
                    bbox=dict(facecolor='white', edgecolor='black', pad=3.0))

    plt.savefig('./'+name)









def plot_single_trace(predicted_xuv, actual_xuv, predicted_ir, actual_ir, trace, reconstructed_trace, name,
                      plot_streaking_mse):

    # initialize the plot for single trace
    fig = plt.figure(figsize=(10,10))
    borderspace_lr = 0.1
    borderspace_tb = 0.05

    plt.subplots_adjust(left=0+borderspace_lr, right=1-borderspace_lr,
                        top=1-borderspace_tb, bottom=0+borderspace_tb,
                        hspace=0.4, wspace=0.4)
    gs = fig.add_gridspec(4, 2)

    # generate time for plotting
    si_time = crab_tf2.tau_values * sc.physical_constants['atomic unit of time'][0]

    # predictions = sess.run(y_pred, feed_dict={x: x_in, y_true: y_in})
    # mse = sess.run(loss, feed_dict={x: x_in[index].reshape(1, -1), y_true: y_in[index].reshape(1, -1)})



    if actual_ir is not None:

        # plotting Actual IR
        crop_phase_right = 1
        crop_phase_left = 1
        axis = fig.add_subplot(gs[0, 0])
        axis.cla()
        axis.plot(ir_fmat_hz * 1e-14, np.abs(actual_ir) ** 2, color="black")
        axtwin = axis.twinx()
        # plot the phase
        phase = np.unwrap(np.angle(actual_ir[crop_phase_left:-crop_phase_right]))
        # set ticks
        tickmax = int(np.max(phase))
        tickmin = int(np.min(phase))
        # ticks = np.arange(0, tickmax + 1, 1)
        # axtwin.set_yticks(ticks)
        phase = phase - np.min(phase)
        axtwin.plot(ir_fmat_hz[crop_phase_left:-crop_phase_right] * 1e-14, phase, color="green", linewidth=3)
        axtwin.set_ylabel(r"$\phi_{IR}$[rad]")
        axtwin.yaxis.label.set_color('green')
        axtwin.tick_params(axis='y', colors='green')
        axtwin.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.text(0, 1.05, "a) Actual IR", transform=axis.transAxes, backgroundcolor='white', weight='bold')
        axis.set_xlabel('Frequency [$10^{14}$Hz]')
        # axis.set_ylabel('$|E_{XUV}(eV)|$')
        axis.set_ylabel('Intensity [arbitrary units]')




    if actual_xuv is not None:
        # plot the actual xuv spectral phase
        # determine xuv phase crop
        crop_phase_right = global_xuv_phase_crop_right
        crop_phase_left = global_xuv_phase_crop_left
        axis = fig.add_subplot(gs[0, 1])
        axis.cla()
        axis.plot(electronvolts, np.abs(actual_xuv) ** 2, color="black")
        axtwin = axis.twinx()
        # plot the phase
        phase = np.unwrap(np.angle(actual_xuv[crop_phase_left:-crop_phase_right]))
        # set ticks
        tickmax = int(np.max(phase))
        tickmin = int(np.min(phase))
        # ticks = np.arange(0, tickmax + 1, 1)
        # axtwin.set_yticks(ticks)
        phase = phase - np.min(phase)
        axtwin.plot(electronvolts[crop_phase_left:-crop_phase_right], phase, color="green", linewidth=3)
        axtwin.set_ylabel(r"$\phi_{XUV}$[rad]")
        axtwin.yaxis.label.set_color('green')
        axtwin.tick_params(axis='y', colors='green')
        axtwin.yaxis.set_major_locator(MaxNLocator(integer=True))
        axis.text(0, 1.05, "b) Actual XUV", transform=axis.transAxes, backgroundcolor='white', weight='bold')
        axis.set_xlabel('Photon Energy [eV]')
        # axis.set_ylabel('$|E_{XUV}(eV)|$')
        axis.set_ylabel('Intensity [arbitrary units]')


    # plot input trace
    axis = fig.add_subplot(gs[1, :])
    axis.pcolormesh(si_time * 1e15, crab_tf2.eV_values, trace, cmap='jet')
    axis.set_xlabel('Time Delay [fs]')
    axis.set_ylabel('Energy [eV]')
    axis.text(0.5, 1.06, "Input Streaking Trace", transform=axis.transAxes, backgroundcolor='white', weight='bold',
              horizontalalignment='center')
    # axis.set_xticks([-15, -10, -5, 0, 5, 10, 15])
    # subplot just for the letter for input trace
    axis.text(0, 1.06, "c)", transform=axis.transAxes, backgroundcolor='white', weight='bold')



    # plot the predicted xuv spectral phase
    # determine xuv phase crop
    crop_phase_right = global_xuv_phase_crop_right
    crop_phase_left = global_xuv_phase_crop_left
    axis = fig.add_subplot(gs[2,1])
    axis.cla()
    axis.plot(electronvolts, np.abs(predicted_xuv)**2, color="black")
    axtwin = axis.twinx()
    # plot the phase
    phase = np.unwrap(np.angle(predicted_xuv[crop_phase_left:-crop_phase_right]))
    phase = phase - np.min(phase)
    axtwin.plot(electronvolts[crop_phase_left:-crop_phase_right],phase, color="green", linewidth=3)
    # set ticks
    tickmax = int(np.max(phase))
    tickmin = int(np.min(phase))
    #ticks = np.arange(0, tickmax + 1, 5)
#    axtwin.set_yticks(ticks)

    axtwin.set_ylabel(r"$\phi_{XUV}$[rad]")
    axtwin.yaxis.label.set_color('green')
    axtwin.tick_params(axis='y', colors='green')
    axtwin.yaxis.set_major_locator(MaxNLocator(integer=True))
    # plot the error
    #mse between xuv traces
    if actual_xuv is not None:
        xuv_mse = calc_mse(predicted_xuv, actual_xuv)
        axtwin.text(0.03, 0.90, "MSE: " + str(round(xuv_mse, 5)), transform=axtwin.transAxes, backgroundcolor='white',
                    bbox=dict(facecolor='white', edgecolor='black', pad=3.0))
    axis.text(0, 1.05, "e) Predicted XUV", transform=axis.transAxes, backgroundcolor='white', weight='bold')
    axis.set_xlabel('Photon Energy [eV]')
    axis.set_ylabel('Intensity [arbitrary units]')


    # plotting Predicted IR
    # determine xuv phase crop
    crop_phase_right = 1
    crop_phase_left = 1
    axis = fig.add_subplot(gs[2, 0])
    axis.cla()
    axis.plot(ir_fmat_hz * 1e-14, np.abs(predicted_ir) ** 2, color="black")
    axtwin = axis.twinx()
    # plot the phase
    phase = np.unwrap(np.angle(predicted_ir[crop_phase_left:-crop_phase_right]))
    # set ticks
    tickmax = int(np.max(phase))
    tickmin = int(np.min(phase))
    # ticks = np.arange(0, tickmax + 1, 1)
    # axtwin.set_yticks(ticks)
    phase = phase - np.min(phase)
    axtwin.plot(ir_fmat_hz[crop_phase_left:-crop_phase_right] * 1e-14, phase, color="green", linewidth=3)
    axtwin.set_ylabel(r"$\phi_{IR}$[rad]")
    axtwin.yaxis.label.set_color('green')
    axtwin.tick_params(axis='y', colors='green')
    axtwin.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.text(0, 1.05, "d) Predicted IR", transform=axis.transAxes, backgroundcolor='white', weight='bold')
    if actual_ir is not None:
        ir_mse = calc_mse(predicted_ir, actual_ir)
        axtwin.text(0.03, 0.90, "MSE: " + str(round(ir_mse, 5)), transform=axtwin.transAxes, backgroundcolor='white',
                    bbox=dict(facecolor='white', edgecolor='black', pad=3.0))
    axis.set_xlabel('Frequency [$10^{14}$Hz]')
    # axis.set_ylabel('$|E_{XUV}(eV)|$')
    axis.set_ylabel('Intensity [arbitrary units]')


    # plotting generated trace
    axis = fig.add_subplot(gs[3, :])
    axis.pcolormesh(si_time * 1e15, crab_tf2.eV_values,
                    reconstructed_trace, cmap='jet')
    axis.set_xlabel('Time Delay [fs]')
    axis.set_ylabel('Energy [eV]')
    axis.text(0.5, 1.05, "Reconstructed Streaking Trace", transform=axis.transAxes, backgroundcolor='white',
              weight='bold', horizontalalignment='center')
    # subplot just for the letter for input trace
    axis.text(0, 1.05, "f)", transform=axis.transAxes, backgroundcolor='white', weight='bold')


    # plot trace MSE
    if plot_streaking_mse:
        trace_reconstructed_reshaped = reconstructed_trace.reshape(-1)
        trace_actual_reshape = trace.reshape(-1)
        trace_mse = (1/len(trace_actual_reshape)) * np.sum((trace_reconstructed_reshaped - trace_actual_reshape)**2)
        print('trace_mse: ', trace_mse)
        axis.text(0.012, 0.90, "MSE: " + str(round(trace_mse, 5)), transform=axis.transAxes, backgroundcolor='white',
                    bbox=dict(facecolor='white', edgecolor='black', pad=3.0))

    plt.savefig('./'+name)


def plot_predictions(traces, xuv_f_preds, xuv_f_actuals, name):

    fig2, axis = plt.subplots(3, 4, figsize=(10, 10))


    # for ax, index in zip([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]):
    # for ax, index in zip([0, 1, 2, 3, 4], [0, 1, 2, 8, 10]):
    for ax, index in zip([0, 1, 2, 3], [0, 1, 2, 3]):

        # generate time for plotting
        si_time = crab_tf2.tau_values * sc.physical_constants['atomic unit of time'][0]

        # plot  actual trace
        axis[0][ax].pcolormesh(si_time*1e15, crab_tf2.eV_values, traces[index], cmap='jet')
        if ax == 0:
            axis[0][ax].set_ylabel('Energy [eV]')
        axis[0][ax].set_xlabel('Time Delay [fs]')
        axis[0][ax].text(0.5, 1.05, 'Streaking Trace {}'.format(str(ax+1)), transform=axis[0][ax].transAxes, horizontalalignment='center',
                         weight='bold')


        # set the number of points to crop when plotting the phase to redice the noise
        crop_phase_right = global_xuv_phase_crop_right
        crop_phase_left = global_xuv_phase_crop_left
        # plot E(t) retrieved
        axis[2][ax].cla()
        axis[2][ax].plot(electronvolts, np.abs(xuv_f_preds[index])**2, color="black")
        axtwin = axis[2][ax].twinx()
        phase = np.unwrap(np.angle(xuv_f_preds[index]))[crop_phase_left:-crop_phase_right]
        phase = phase - np.min(phase)
        axtwin.plot(electronvolts[crop_phase_left:-crop_phase_right], phase, color="green", linewidth=3)
        # set ticks
        #tickmax = int(np.max(phase))
        #tickmin = int(np.min(phase))
        #ticks = np.arange(tickmin, tickmax+1, 1)
        #axtwin.set_yticks(ticks)
        axtwin.yaxis.set_major_locator(MaxNLocator(integer=True))
        axtwin.yaxis.label.set_color('green')
        axtwin.tick_params(axis='y', colors='green')
        # plot the error
        mse = calc_mse(xuv_f_preds[index], xuv_f_actuals[index])
        axtwin.text(0, 0.95, "MSE: " + str(round(mse, 5)), transform=axtwin.transAxes, backgroundcolor='white', bbox=dict(facecolor='white', edgecolor='black', pad=3.0))
        axis[2][ax].set_xlabel('Photon Energy [eV]')
        axis[2][ax].text(0.5, 1.05, 'Prediction {}'.format(str(ax+1)), transform=axis[2][ax].transAxes, horizontalalignment='center',
                         weight='bold')
        if ax == 0:
            axis[2][ax].set_ylabel('Intensity [arbitrary units]')
        if ax == 3:
            axtwin.set_ylabel('$\phi_{XUV}$[rad]')



        # plot E(t) actual
        axis[1][ax].cla()
        # complex_field = y_in[index, :64] + 1j * y_in[index, 64:]
        axis[1][ax].plot(electronvolts, np.abs(xuv_f_actuals[index])**2, color="black")
        axis[1][ax].text(0.5,1.05,'Actual {}'.format(str(ax+1)), transform=axis[1][ax].transAxes, horizontalalignment='center', weight='bold')
        axtwin = axis[1][ax].twinx()
        phase = np.unwrap(np.angle(xuv_f_actuals[index]))[crop_phase_left:-crop_phase_right]
        phase = phase - np.min(phase)
        axtwin.plot(electronvolts[crop_phase_left:-crop_phase_right], phase, color="green", linewidth=3)
        # set ticks
        tickmax = int(np.max(phase))
        tickmin = int(np.min(phase))
        ticks = np.arange(tickmin, tickmax+1, 1)
        # axtwin.set_yticks(ticks)
        axtwin.yaxis.set_major_locator(MaxNLocator(integer=True))
        axtwin.yaxis.label.set_color('green')
        axtwin.tick_params(axis='y', colors='green')
        # axis[1][ax].text(0.1, 1, "actual [" + set + " set]", transform=axis[1][ax].transAxes, backgroundcolor='white')
        axis[1][ax].set_xlabel('Photon Energy [eV]')
        if ax == 0:
            axis[1][ax].set_ylabel('Intensity [arbitrary units]')
        if ax == 3:
            axtwin.set_ylabel('$\phi_{XUV}$[rad]')
        if ax != 0:
            axis[0][ax].set_yticks([])
            axis[1][ax].set_yticks([])
            axis[2][ax].set_yticks([])
        # set y limits to the same
        axis[1][ax].set_ylim(-0.05, 1.05)
        axis[2][ax].set_ylim(-0.05, 1.05)
        # axis[1][ax].set_xticks([100, 350, 600])
        # axis[2][ax].set_xticks([100, 350, 600])


    # print("mses: ", mses)
    # print("avg : ", (1 / len(mses)) * np.sum(np.array(mses)))

    # save image
    outerwidth = 0.06
    plt.subplots_adjust(left=outerwidth, right=1-outerwidth, top=0.96, bottom=0.05,wspace=0.2, hspace=0.4)
    plt.savefig('./'+name)
    # dir = "/home/zom/PythonProjects/attosecond_streaking_phase_retrieval/nnpictures/" + modelname + "/" + set + "/"
    # if not os.path.isdir(dir):
    #     os.makedirs(dir)
    # fig.savefig(dir + str(epoch) + ".png")


def retrieve_pulse(filepath, plotting=False):
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        matrix = np.array(list(reader))

        Energy = matrix[4:, 0].astype('float')
        Delay = matrix[2, 2:].astype('float')
        Values = matrix[4:, 2:].astype('float')

    # map the function onto a
    interp2 = interpolate.interp2d(Delay, Energy, Values, kind='linear')

    delay_new = np.linspace(Delay[0], Delay[-1], 176)
    energy_new = np.linspace(Energy[0], Energy[-1], 200)

    values_new = interp2(delay_new, energy_new)

    if plotting:

        fig = plt.figure()
        gs = fig.add_gridspec(2, 2)

        ax = fig.add_subplot(gs[0, :])
        ax.pcolormesh(Delay, Energy, Values, cmap='jet')
        ax.set_xlabel('fs')
        ax.set_ylabel('eV')
        ax.text(0.1, 0.9, 'original', transform=ax.transAxes, backgroundcolor='white')

        ax = fig.add_subplot(gs[1, :])
        ax.pcolormesh(delay_new, energy_new, values_new, cmap='jet')
        ax.set_xlabel('fs')
        ax.set_ylabel('eV')
        ax.text(0.1, 0.9, 'interpolated', transform=ax.transAxes, backgroundcolor='white')

        plt.show()

    return delay_new, energy_new, values_new


def get_measured_trace():
    filepath = './experimental_data/53asstreakingdata.csv'
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        matrix = np.array(list(reader))

        Energy = matrix[4:, 0].astype('float')
        Delay = matrix[2, 2:].astype('float')
        Values = matrix[4:, 2:].astype('float')


    # map the function onto a grid matching the training data
    interp2 = interpolate.interp2d(Delay, Energy, Values, kind='linear')
    timespan = np.abs(Delay[-1]) + np.abs(Delay[0])
    # make it 200 Energy, 80 delay
    delay_new = np.arange(Delay[0], Delay[-1], timespan/80)
    energy_new = np.linspace(Energy[0], Energy[-1], 200)


    values_new = interp2(delay_new, energy_new)

    # interpolate to momentum [a.u]
    energy_new_joules = energy_new * sc.electron_volt # joules
    energy_new_au = energy_new_joules / sc.physical_constants['atomic unit of energy'][0]  # a.u.
    momentum_new_au = np.sqrt(2 * energy_new_au)
    interp2_momentum = interpolate.interp2d(delay_new, momentum_new_au, values_new, kind='linear')

    # interpolate onto linear momentum axis
    N = len(momentum_new_au)
    momentum_linear = np.linspace(momentum_new_au[0], momentum_new_au[-1], N)
    values_lin_momentum = interp2_momentum(delay_new, momentum_linear)


    return delay_new, momentum_linear, values_lin_momentum


def get_trace(index, filename):

    with tables.open_file(filename, mode='r') as hdf5_file:

        trace = hdf5_file.root.trace[index,:]

        actual_fields = {}

        # actual_fields['ir_t'] = hdf5_file.root.ir_t[index,:]
        actual_fields['ir_f'] = hdf5_file.root.ir_f[index,:]

        # actual_fields['xuv_t'] = hdf5_file.root.xuv_t[index, :]
        actual_fields['xuv_f'] = hdf5_file.root.xuv_f[index, :]




    return trace.reshape(1,-1), actual_fields



if __name__ == "__main__":

    global_xuv_phase_crop_right = 35
    global_xuv_phase_crop_left = 10


    modelname = 'kspace_measured_spectrum2_5coefs_multiresmoreweights_matchmeasuredgridspace_dynamicL'

    # get the trace
    trace, actual_fields = get_trace(index=3, filename='attstrace_test2_processed.hdf5')

    # take a field to get the spectrum
    spectrum_measured = np.abs(actual_fields['xuv_f'])**2






    # get trace from experimental
    # _, _, trace = get_measured_trace()
    # trace = trace.reshape(1, -1)
    # actual_fields = None

    with tf.Session() as sess:

        saver = tf.train.Saver()
        saver.restore(sess, './models/{}.ckpt'.format(modelname))

        # frequency axis for plotting xuv
        # MAKE SURE THIS XUV IS XUV USED IN NETWORK
        fmat = crab_tf2.xuv.f_cropped# a.u.
        fmat_hz = fmat / sc.physical_constants['atomic unit of time'][0]
        fmat_joules = sc.h * fmat_hz  # joules
        electronvolts = 1 / (sc.elementary_charge) * fmat_joules

        # frequency axis for plotting ir
        ir_f = crab_tf2.ir.f_cropped
        ir_fmat_hz = ir_f / sc.physical_constants['atomic unit of time'][0]





        # generate an image from the input image
        generated_image = sess.run(network2.image, feed_dict={network2.x: trace})
        trace_2d = trace.reshape(len(crab_tf2.p_values), len(crab_tf2.tau_values))
        predicted_fields_vector = sess.run(network2.y_pred, feed_dict={network2.x: trace})

        predicted_fields = {}
        predicted_fields['xuv_f'], predicted_fields['ir_f'] = network2.separate_xuv_ir_vec(predicted_fields_vector[0])

        # plot_single_trace(predicted_xuv=predicted_fields['xuv_f'],
        #                   actual_xuv=actual_fields['xuv_f'],
        #                   predicted_ir=predicted_fields['ir_f'],
        #                   actual_ir=actual_fields['ir_f'],
        #                   trace=trace_2d,
        #                   reconstructed_trace=generated_image,
        #                   name='fromtestset.png',
        #                   plot_streaking_mse=False
        #                   )

        plot_single_trace_xuv_only_square(predicted_xuv=predicted_fields['xuv_f'],
                          actual_xuv=actual_fields['xuv_f'],
                          trace=trace_2d,
                          reconstructed_trace=generated_image,
                          name='fromtestset.png',
                          plot_streaking_mse=True
                          )





        traces = []
        xuv_f_preds = []
        xuv_f_actuals = []

        for index in [1, 9, 3, 4]:

            trace, actual_fields = get_trace(index=index, filename='attstrace_test2_processed.hdf5')
            trace_2d = trace.reshape(len(crab_tf2.p_values), len(crab_tf2.tau_values))
            predicted_fields_vector = sess.run(network2.y_pred, feed_dict={network2.x: trace})
            xuv_f_pred, _ = network2.separate_xuv_ir_vec(predicted_fields_vector[0])

            traces.append(trace_2d)
            xuv_f_preds.append(xuv_f_pred)
            xuv_f_actuals.append(actual_fields['xuv_f'])

        plot_predictions(traces=traces, xuv_f_preds=xuv_f_preds, xuv_f_actuals=xuv_f_actuals,
                         name='predictions.png')


        #plot the experimental retrieval
        with open('importantstuff.p', 'rb') as file:
            data = pickle.load(file)

            print('xuv length: ')
            print(len(data['predicted_fields']['xuv_f']))

            print('ir length: ')
            print(len(data['predicted_fields']['ir_f']))






        # plot_single_trace(predicted_xuv=data['predicted_fields']['xuv_f'],
        #                   actual_xuv=None,
        #                   predicted_ir=data['predicted_fields']['ir_f'],
        #                   actual_ir=None,
        #                   trace=data['input_image'],
        #                   reconstructed_trace=data['generated_image'],
        #                   name='experimental.png',
        #                   plot_streaking_mse=True
        #                   )

        plot_single_trace_xuv_only_measured(predicted_xuv=data['predicted_fields']['xuv_f'],
                          actual_xuv=None,
                          trace=data['input_image'],
                          reconstructed_trace=data['generated_image'],
                          name='experimental.png',
                          plot_streaking_mse=True,
                          measured_spec=spectrum_measured
                          )

        # plot the steps in time
        filenames = []
        filenames.append(
            './run_kspace_measured_spectrum2_5coefs_multiresmoreweights_matchmeasuredgridspace_dynamicL-tag-train_mse.csv')
        #filenames.append(
            #'./run_kspace_measured_spectrum2_5coefs_multiresmoreweights_matchmeasuredgridspace_dynamicL-tag-test_mse.csv')
        plot_mse_steps(filenames, name='traindata.png')









        plt.show()














