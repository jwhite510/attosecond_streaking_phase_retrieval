import numpy as np
import matplotlib.pyplot as plt


def retrieve_trace():
    trace = []
    for line in open("53as_trace.dat", "r"):
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

    # normalize trace
    trace = trace / np.max(trace)

    return delay, energy, trace


if __name__ == "__main__":

    delay, energy, trace = retrieve_trace()

    plt.figure(1)
    plt.pcolormesh(delay, energy, trace, cmap="jet")

    plt.show()




