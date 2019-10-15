import numpy as np
import matplotlib.pyplot as plt
import phase_parameters.params


def save_arrays(arr_x, arr_y, name):
    mat = np.append(arr_x.reshape(-1,1), arr_y.reshape(-1,1), axis=1)
    np.savetxt(name, mat, fmt="%.8f", delimiter=",")


if __name__ == "__main__":

    # make electron spectrum
    x_electron = np.linspace(50.0, 350.0, 100)
    width = 700.0
    electron_spec = np.exp(-(( x_electron-150.0 )**2)/width)

    # make cross section
    cross_section_ev = np.linspace(50, 400, 300)
    # cross_section = 1 / cross_section_ev
    cross_section = np.ones_like(cross_section_ev)
    cross_section = cross_section / np.max(cross_section)

    # plt.figure(1)
    # plt.plot(cross_section_ev, cross_section)
    # plt.show()
    # exit()



    # save the electron spectrum
    save_arrays(x_electron, electron_spec, "xuv_spectrum/sample4/spectrum4_electron_gen.csv")
    save_arrays(cross_section_ev, cross_section, "xuv_spectrum/sample4/HeliumCrossSection_gen.csv")

    # save the cross section
    # save the measred trace energy values

    plt.figure(1)
    plt.plot(x_electron, electron_spec, label="electron spectrum")
    plt.plot(x_electron+phase_parameters.params.Ip_eV, electron_spec, label=r"+ $I_p$")
    plt.plot(cross_section_ev, cross_section, label="cross_section")
    plt.gca().legend()
    plt.show()
