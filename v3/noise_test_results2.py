import numpy as np
import matplotlib.pyplot as plt
import pickle
import unsupervised_retrieval
import xuv_spectrum.spectrum as xuvspec


def get_rmses(trace_type, actual_phase):

    with open("noise_test7__.p", "rb") as file:
        obj = pickle.load(file)

    # order the dictionary
    count_numbers = []
    contents = []
    for key in obj:
        if key != "actual_values":
            count_numbers.append(key)
            contents.append(obj[key])


    phase_error = {"nn": [], "nn_init": [], "ga": []}
    trace_rmse = {"nn": [], "nn_init": [], "ga": []}

    for count, content in zip(count_numbers, contents):

        this_phase_error = calculate_phase_error(actual_phase, content[trace_type]["nn"]["field"]["cropped_phase"])
        phase_error["nn"].append(this_phase_error)
        this_trace_rmse = np.sqrt(content[trace_type]["nn"]["trace"]["mse"])
        trace_rmse["nn"].append(this_trace_rmse)

        this_phase_error = calculate_phase_error(actual_phase, content[trace_type]["nn_init"]["field"]["cropped_phase"])
        phase_error["nn_init"].append(this_phase_error)
        this_trace_rmse = np.sqrt(content[trace_type]["nn_init"]["trace"]["mse"])
        trace_rmse["nn_init"].append(this_trace_rmse)

        this_phase_error = calculate_phase_error(actual_phase, content[trace_type]["ga"]["field"]["cropped_phase"])
        phase_error["ga"].append(this_phase_error)
        this_trace_rmse = np.sqrt(content[trace_type]["ga"]["trace"]["mse"])
        trace_rmse["ga"].append(this_trace_rmse)

    count_numbers = [float(num) for num in count_numbers]
    snrs = np.sqrt(count_numbers)

    fig = plt.figure(figsize=(7,7))
    gs = fig.add_gridspec(2,1)

    ax = fig.add_subplot(gs[0,0])
    ax.plot(snrs, phase_error["nn_init"], label="Network Initial Output (normal trace)", color="blue")
    ax.plot(snrs, phase_error["nn"], label="Unsupervised Learning", color="red")
    ax.plot(snrs, phase_error["ga"], label="Genetic Algorithm", color="green")
    ax.set_title(trace_type + " trace")
    ax.set_ylabel("Retrieved Phase Error")
    ax.set_xlabel("Signal to Noise Ratio")
    ax.legend()

    ax = fig.add_subplot(gs[1,0])
    ax.plot(snrs, trace_rmse["nn_init"], label="Network Initial Output (normal trace)", color="blue")
    ax.plot(snrs, trace_rmse["nn"], label="Unsupervised Learning", color="red")
    ax.plot(snrs, trace_rmse["ga"], label="Genetic Algorithm", color="green")
    ax.set_ylabel("Trace RMSE")
    ax.set_xlabel("Signal to Noise Ratio")
    ax.legend()

    fig.savefig(trace_type+".png")


def calculate_phase_error(phase_actual, phase_measured):

    # test mse
    # print("test mse:")
    # print(unsupervised_retrieval.calculate_rmse(phase_actual, phase_measured))

    # get the intensity
    Ef_cropped = xuvspec.Ef[xuvspec.indexmin:xuvspec.indexmax]
    Intensity = np.abs(Ef_cropped)**2
    N = len(Intensity)
    I_max = np.max(Intensity)

    # phase difference:
    phi_diff = np.abs(phase_actual - phase_measured)

    # summation of weighted difference
    summation = np.sum(phi_diff * Intensity)

    # divide by n and normalize
    error = summation / (N * I_max)

    return error


if __name__ == "__main__":

    _, actual_phase, _ = unsupervised_retrieval.get_fake_measured_trace(counts=100, plotting=False, run_name=None)

    get_rmses("proof", actual_phase)
    get_rmses("normal", actual_phase)
    get_rmses("autocorrelation", actual_phase)
    plt.show()






