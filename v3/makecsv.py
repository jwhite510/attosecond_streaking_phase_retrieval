import numpy as np
import matplotlib.pyplot as plt
# import phase_parameters.params


def save_array_1d(arr, name):
    np.savetxt(name, arr, fmt="%.8f")

def save_arrays(arr_x, arr_y, name):
    mat = np.append(arr_x.reshape(-1,1), arr_y.reshape(-1,1), axis=1)
    np.savetxt(name, mat, fmt="%.8f", delimiter=",")


e_0 = 20.0

if __name__ == "__main__":

    # make electron spectrum
    x_electron = np.linspace(5.0, 350.0, 100)
    width = 700.0
    electron_spec = np.exp(-(( x_electron-e_0 )**2)/width)

    # multiply by a rec to make a sharp cutoff
    rect = np.ones_like(electron_spec)
    rect[x_electron<(e_0-10)] = 0
    electron_spec = electron_spec * rect

    # make cross section
    cross_section_ev = np.linspace(10, 400, 300)
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
    energy_shift = -40
    trace_energy_vals_ev = np.arange(50.0+energy_shift, 351.0+energy_shift, 1.0)
    save_array_1d(trace_energy_vals_ev, "measured_trace/sample4/energy_gen.csv")


    # save the cross section
    # save the measred trace energy values

    plt.figure(1)
    plt.plot(x_electron, electron_spec, label="electron spectrum")
    plt.plot(x_electron+24.587, electron_spec, label=r"+ $I_p$")
    plt.plot(cross_section_ev, cross_section, label="cross_section")
    plt.plot(trace_energy_vals_ev, 1.1*np.ones_like(trace_energy_vals_ev), label="trace plot $eV$")
    plt.gca().legend()
    plt.show()
