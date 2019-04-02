import numpy as np
import matplotlib.pyplot as plt
import os
import csv


def retrieve_trace3():
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

    return delay, energy, trace


def retrieve_trace2():

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
    # Dtau = Delay_even[-1] - Delay_even[-2]
    # print('Delay: ', Delay)
    # print('Delay_even: ', Delay_even)
    # print('np.shape(values_even): ', np.shape(values_even))
    # print('len(values_even.reshape(-1))', len(values_even.reshape(-1)))
    # print('Dtau: ', Dtau)
    # print('Delay max', Delay_even[-1])
    # print('N: ', len(Delay_even))
    # print('Energy: ', len(Energy))
    # f0 = find_central_frequency_from_trace(trace=values_even, delay=Delay_even, energy=Energy)
    # print(f0)  # in seconds
    # lam0 = sc.c / f0
    # print('f0 a.u.: ', f0 * sc.physical_constants['atomic unit of time'][0])  # convert f0 to atomic unit
    # print('lam0: ', lam0)

    # normalize values
    return Delay_even, Energy, values_even


trace_num = 3


if trace_num == 2:

    delay, energy, trace = retrieve_trace2()

elif trace_num == 3:

    delay, energy, trace = retrieve_trace3()



if __name__ == "__main__":

    delay, energy, trace = retrieve_trace3()

    plt.figure(1)
    plt.pcolormesh(delay, energy, trace, cmap="jet")

    plt.show()




