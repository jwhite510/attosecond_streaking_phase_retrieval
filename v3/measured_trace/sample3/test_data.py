import numpy as np
import matplotlib.pyplot as plt


def retrieve_trace():
    mat = []
    for line in open("53as_trace.dat", "r"):
        line = line.rstrip()
        line = line.split("\t")
        line = [float(e) for e in line]
        mat.append(line)
    mat = np.transpose(np.array(mat))

    # delay_values =



    # normalize mat
    mat = mat / np.max(mat)

    return mat




if __name__ == "__main__":

    trace = retrieve_trace()

    plt.figure(1)
    plt.pcolormesh(trace, cmap="jet")

    plt.show()




