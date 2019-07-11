import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import scipy.constants as sc


def find_f0(x, y):

    x = np.array(x)
    y = np.array(y)

    maxvals = []

    for _ in range(3):
        max_index = np.argmax(y)
        maxvals.append(x[max_index])

        x = np.delete(x, max_index)
        y = np.delete(y, max_index)

    maxvals = np.delete(maxvals, np.argmin(np.abs(maxvals)))

    return maxvals[np.argmax(maxvals)]


def find_central_frequency_from_trace(trace, delay, energy, plotting=False):

    # make sure delay is even
    assert len(delay) % 2 == 0

    N = len(delay)
    # print('N: ', N)
    dt = delay[-1] - delay[-2]
    df = 1 / (dt * N)
    freq_even = df * np.arange(-N / 2, N / 2)
    # plot the streaking trace and ft

    trace_f = np.fft.fftshift(np.fft.fft(np.fft.fftshift(trace, axes=1), axis=1), axes=1)

    # summation along vertical axis
    integrate = np.sum(np.abs(trace_f), axis=0)

    # find the maximum values
    f0 = find_f0(x=freq_even, y=integrate)  # seconds

    lam0 = sc.c / f0

    if plotting:
        # find central frequency
        _, ax = plt.subplots(3, 1)
        ax[0].pcolormesh(delay, energy, trace, cmap='jet')
        ax[1].pcolormesh(freq_even, energy, np.abs(trace_f), cmap='jet')
        ax[2].plot(freq_even, integrate)

    return f0, lam0


def retrieve_trace3(find_f0=False):
    trace = []

    for line in open(os.path.dirname(__file__)+"/sample3/53as_trace.dat", "r"):
        line = line.rstrip()
        line = line.split("\t")
        line = [float(e) for e in line]
        trace.append(line)
    trace = np.transpose(np.array(trace))

    delay_min = -5.47 #fs
    delay_max = 5.44 #fs
    delay = np.linspace(delay_min, delay_max, np.shape(trace)[1])

    e_min = 50
    e_max = 350
    energy = np.linspace(e_min, e_max, np.shape(trace)[0])

    # remove the last delay step so the delay axis is an even number for fourier transform
    trace = trace[:, :-1]
    delay = delay[:-1]

    # convert to seconds
    delay = delay * 1e-15

    # normalize trace
    trace = trace / np.max(trace)

    if find_f0:
        f0, lam0 = find_central_frequency_from_trace(trace=trace, delay=delay, energy=energy, plotting=True)
        print(f0)  # in seconds
        print(lam0)

    return delay, energy, trace


def retrieve_trace2(find_f0=False):

    filepath = './measured_trace/sample2/MSheet1_1.csv'
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        matrix = np.array(list(reader))

        Energy = matrix[1:, 0].astype('float') # eV
        Delay = matrix[0, 1:].astype('float') # fs
        Values = matrix[1:, 1:].astype('float')

    #print(Delay)
    # print('len(Energy): ', len(Energy))
    # print('Energy: ', Energy)

    # construct frequency axis with even number for fourier transform
    values_even = Values[:, :-1]
    Delay_even = Delay[:-1]
    Delay_even = Delay_even * 1e-15  # convert to seconds

    if find_f0:
        f0, lam0 = find_central_frequency_from_trace(trace=values_even, delay=Delay_even, energy=Energy, plotting=True)
        print(f0)  # in seconds
        print(lam0)

    return Delay_even, Energy, values_even

def retrieve_trace4(find_f0=False):
    trace = []
    electron_volt_vals = []

    # first character of each line is the x energy value
    for line_num, line in enumerate(open(os.path.dirname(__file__)+"/sample4/trace4.csv", "r")):
        if line_num == 0:
            # line.split(",")
            delay_vals = line.replace("\n", "").split(",")[1:]
            delay_vals = [float(e) for e in delay_vals]
        else:
            line = line.rstrip()
            line = line.split(",")
            line = [float(e) for e in line]
            electron_volt_vals.append(line.pop(0))
            trace.append(line)

    electron_volt_vals = np.array(electron_volt_vals)
    delay_vals = np.array(delay_vals)
    trace = np.array(trace)


    # remove the last delay step so the delay axis is an even number for fourier transform
    trace = trace[:, :-1]
    delay_vals = delay_vals[:-1]

    # plt.figure(33)
    # plt.pcolormesh(delay_vals, electron_volt_vals, trace, cmap="jet")
    # plt.show()
    # exit()

    # convert to seconds
    delay_vals = delay_vals * 1e-15

    # normalize trace
    trace = trace / np.max(trace)

    if find_f0:
        f0, lam0 = find_central_frequency_from_trace(trace=trace, delay=delay_vals, energy=electron_volt_vals, plotting=True)
        print(f0)  # in seconds
        print(lam0)# 1.6788377648000736e-06

    return delay_vals, electron_volt_vals, trace



trace_num = 4


if trace_num == 2:

    delay, energy, trace = retrieve_trace2()

elif trace_num == 3:

    delay, energy, trace = retrieve_trace3()

elif trace_num == 4:

    delay, energy, trace = retrieve_trace4()


if __name__ == "__main__":

    delay, energy, trace = retrieve_trace3(find_f0=True)

    plt.figure()
    plt.pcolormesh(delay, energy, trace, cmap="jet")

    plt.show()




