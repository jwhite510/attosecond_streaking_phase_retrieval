import numpy as np
import matplotlib.pyplot as plt


def get_values(filename):

    with open(filename, "r") as file:
        file_contents = list(file.readlines())
        epochs = []
        error = []
        for line in file_contents[1:]:
            epochs.append(float(line.strip('\n').split(',')[1]))
            error.append(float(line.strip('\n').split(',')[2]))

        return error, epochs

# plot the 





# 1. fields MSE
fig, ax = plt.subplots(4, 1, figsize=(7,10))
fig.subplots_adjust(left=0.2, hspace=0.0, right=0.7)
error, epochs = get_values("run_plot_retrieval_individualcoefs-tag-train_mse_fields.csv")
ax[0].plot(epochs, error, color="red", label="train", linestyle="dashed")

error, epochs = get_values("run_plot_retrieval_individualcoefs-tag-test_mse_fields.csv")
ax[0].plot(epochs, error, color="red", label="validation")

ax[0].set_ylabel("Complex Fields\n (IR + XUV)\nMSE Error")
ax[0].legend(bbox_to_anchor=(1.3, 0.5), loc="center")
ax[0].set_yscale("log")
ax[0].set_xticks([])



# 2. phasecurve MSE
error, epochs = get_values("run_plot_retrieval_individualcoefs-tag-train_mse_phasecurve.csv")
ax[1].plot(epochs, error, color="red", label="train", linestyle="dashed")

error, epochs = get_values("run_plot_retrieval_individualcoefs-tag-test_mse_phasecurve.csv")
ax[1].plot(epochs, error, color="red", label="validation")

ax[1].set_ylabel("Phase Curve\n Vector (XUV)\nMSE Error")
ax[1].legend(bbox_to_anchor=(1.3, 0.5), loc="center")
ax[1].set_yscale("log")
ax[1].set_xticks([])


# 3. coefficients MSE
# total avg coefficietts
error, epochs = get_values("run_plot_retrieval_individualcoefs-tag-test_mse_coef_params.csv")
ax[2].plot(epochs, error, color="red", label="both/validation")
error, epochs = get_values("run_plot_retrieval_individualcoefs-tag-train_mse_coef_params.csv")
ax[2].plot(epochs, error, color="red", label="both/train", linestyle="dashed")
# just the xuv coefs
error, epochs = get_values("run_plot_retrieval_individualcoefs-tag-xuv_coefs_avg_test.csv")
ax[2].plot(epochs, error, color="blue", label="xuv/validation")
error, epochs = get_values("run_plot_retrieval_individualcoefs-tag-xuv_coefs_avg_train.csv")
ax[2].plot(epochs, error, color="blue", label="xuv/train", linestyle="dashed")
# just the ir params
error, epochs = get_values("run_plot_retrieval_individualcoefs-tag-ir_avg_test.csv")
ax[2].plot(epochs, error, color="purple", label="ir/validation")
error, epochs = get_values("run_plot_retrieval_individualcoefs-tag-ir_avg_train.csv")
ax[2].plot(epochs, error, color="purple", label="ir/train", linestyle="dashed")

ax[2].set_ylabel("XUV Coefficients/IR\n Params\nMSE Error")
ax[2].legend(bbox_to_anchor=(1.3, 0.5), loc="center")
ax[2].set_xticks([])
ax[2].set_yscale("log")

# 4. individual XUV coefficients
for i, color in zip([1, 2, 3, 4, 5], ["red", "blue", "orange", "pink", "green"]):
    error, epochs = get_values("run_plot_retrieval_individualcoefs-tag-xuv_coef"+str(i)+"_test.csv")
    ax[3].plot(epochs, error, color=color, label="$\phi_{}$/validation".format(str(i)))
    error, epochs = get_values("run_plot_retrieval_individualcoefs-tag-xuv_coef"+str(i)+"_train.csv")
    ax[3].plot(epochs, error, color=color, label="$\phi_{}$/train".format(str(i)), linestyle="dashed")
ax[3].legend(bbox_to_anchor=(1.3, 0.5), loc="center")
ax[3].set_yscale("log")
ax[3].set_ylabel("Individual \nCoefficients (XUV)\nMSE Error")
ax[3].set_xlabel("Epoch")


# add a line to show the training stages
for i in range(4):
    ymin, ymax = ax[i].get_ylim()
    ax[i].plot([15, 15], [ymin, ymax], color="blue", alpha=0.5, linestyle="dashed")
ax[0].text(0.0, 1.1, "Epoch > 15: cost function: Complex Fields\nEpoch < 15 cost function: Phase Coefficients/IR Parameters", transform=ax[0].transAxes)
ax[0].text(0.14, 0.8, "Epoch 15", transform=ax[0].transAxes, color="blue", alpha=0.5)

plt.savefig("./fig.png")
