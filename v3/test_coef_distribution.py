import tables
import numpy as np
import matplotlib.pyplot as plt










if __name__ == "__main__":

    phi_2 = []
    phi_3 = []
    phi_4 = []
    phi_5 = []


    with tables.open_file('train3.hdf5', mode='r') as hd5file:

        # index = 0
        samples = len(hd5file.root.xuv_coefs[:, :])

        for index in range(samples):

            xuv_coefs = hd5file.root.xuv_coefs[index, :]
            phi_2.append(xuv_coefs[1])
            phi_3.append(xuv_coefs[2])
            phi_4.append(xuv_coefs[3])
            phi_5.append(xuv_coefs[4])


    plt.figure(1)
    plt.hist(phi_2, color="blue", width=0.02, bins=50)
    plt.title("$\phi_2$ ")
    plt.xlabel("coef value")
    plt.ylabel("count")
    plt.gca().set_yscale("log")

    plt.figure(2)
    plt.hist(phi_3, color="blue", width=0.02, bins=50)
    plt.title("$\phi_3$ ")
    plt.xlabel("coef value")
    plt.ylabel("count")
    plt.gca().set_yscale("log")

    plt.figure(3)
    plt.hist(phi_4, color="blue", width=0.02, bins=50)
    plt.title("$\phi_4$ ")
    plt.xlabel("coef value")
    plt.ylabel("count")
    plt.gca().set_yscale("log")

    plt.figure(4)
    plt.hist(phi_5, color="blue", width=0.02, bins=50)
    plt.title("$\phi_5$ ")
    plt.xlabel("coef value")
    plt.ylabel("count")
    plt.gca().set_yscale("log")








    plt.show()







