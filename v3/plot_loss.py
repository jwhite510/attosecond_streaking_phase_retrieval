import csv
import numpy as np
import matplotlib.pyplot as plt



def get_csv(filename):
    with open(filename) as file:
        reader = csv.reader(file)
        content = np.array(list(reader))
        data = content[1:].astype(np.float)
    return data



# #---------------------------------------------
# #---------------------------------------------
# # -----------supervised learning---------------
# #---------------------------------------------
# #---------------------------------------------
# fig, ax = plt.subplots(3, 1, figsize=(6,9))
# fig.subplots_adjust(wspace=0.0, hspace=0.0, top=1.0, left=0.1, bottom=0.1)
# data = get_csv(filename="run_test1_phasecurve-tag-train_mse_coef_params.csv")
# ax[0].plot(data[:, 1], data[:, 2], color="blue", label="Coefficient\nParameters MSE train")
# ax[0].plot([15, 15], [0, np.max(data[:, 2])], color="red")
# ax[0].plot([30, 30], [0, np.max(data[:, 2])], color="red")
# ax[0].set_yscale("log")
# ax[0].set_ylim(0, np.max(data[:, 2]))
# ax[0].set_xticks([])
# # plot test data
# data = get_csv(filename="run_test1_phasecurve-tag-test_mse_coef_params.csv")
# ax[0].plot(data[:, 1], data[:, 2], color="orange", label="validation Set")
# # ax[0].set_xlim(0, 60)
# ax[0].legend()
#
#
# data = get_csv(filename="run_test1_phasecurve-tag-train_mse_fields.csv")
# ax[1].plot(data[:, 1], data[:, 2], color="blue", label="Fields\nvector MSE")
# ax[1].plot([15, 15], [0, np.max(data[:, 2])], color="red")
# ax[1].plot([30, 30], [0, np.max(data[:, 2])], color="red")
# ax[1].set_yscale("log")
# ax[1].set_ylim(0, np.max(data[:, 2]))
# ax[1].set_xticks([])
# data = get_csv(filename="run_test1_phasecurve-tag-test_mse_fields.csv")
# ax[1].plot(data[:, 1], data[:, 2], color="orange", label="validation Set")
# # ax[1].set_xlim(0, 60)
# ax[1].legend()
#
#
# data = get_csv(filename="run_test1_phasecurve-tag-train_mse_phasecurve.csv")
# ax[2].plot(data[:, 1], data[:, 2], color="blue", label="phasecurve\nvector MSE")
# ax[2].plot([15, 15], [0, np.max(data[:, 2])], color="red")
# ax[2].plot([30, 30], [0, np.max(data[:, 2])], color="red")
# ax[2].set_yscale("log")
# ax[2].set_ylim(0, np.max(data[:, 2]))
# ax[2].set_xlabel("Epoch")
# data = get_csv(filename="run_test1_phasecurve-tag-test_mse_phasecurve.csv")
# ax[2].plot(data[:, 1], data[:, 2], color="orange", label="validation Set")
# # ax[2].set_xlim(0, 60)
# ax[2].legend()
#
# plt.savefig("./mses_full.png")
#
#
#
# #---------------------------------------------
# #---------------------------------------------
# # -----------unsupervised learning------------
# #---------------------------------------------
# #---------------------------------------------
# fig, ax = plt.subplots(1, 1, figsize=(6,5))
# # fig.subplots_adjust(wspace=0.0, hspace=0.0, top=1.0, left=0.1, bottom=0.1)
# data = get_csv(filename="run_run1_normal-tag-trace_mse.csv")
# ax.plot(data[:, 1], data[:, 2], color="blue", label="Trace MSE")
# ax.set_yscale("log")
# ax.set_xlabel("Epoch")
# ax.set_ylim(0, np.max(data[:, 2]))
# ax.legend()
# plt.savefig("unsupervised_normal_retrieval.png")
#
#
#
# fig, ax = plt.subplots(1, 1, figsize=(6,5))
# # fig.subplots_adjust(wspace=0.0, hspace=0.0, top=1.0, left=0.1, bottom=0.1)
# data = get_csv(filename="run_run1_log_b10_t1-tag-trace_mse.csv")
# ax.plot(data[:, 1], data[:, 2], color="blue", label="$log_{10}(trace+1)$\nTrace MSE")
# ax.set_yscale("log")
# ax.set_xlabel("Epoch")
# ax.set_ylim(0, np.max(data[:, 2]))
# ax.legend()
# plt.savefig("unsupervised_log_retrieval.png")




#-----------------------------------------------------------------
#-----------------------------------------------------------------
#----unsupervised retrieval with proof, autocorrelate, normal-----
#-----------------------------------------------------------------
#-----------------------------------------------------------------
fig, ax = plt.subplots(3, 1, figsize=(6,9))
# fig.subplots_adjust(wspace=0.0, hspace=0.0, top=1.0, left=0.1, bottom=0.1)
data = get_csv(filename="run_3Anormal-tag-trace_mse.csv")
ax[0].plot(data[:, 1], data[:, 2], color="blue", label="Normal Trace Retrieval\nMSE")
ax[0].set_yscale("log")
ax[0].set_ylim(0, np.max(data[:, 2]))
ax[0].set_xticks([])
ax[0].set_xlim(0, 2000)
ax[0].legend()


data = get_csv(filename="run_3Aproof-tag-trace_mse.csv")
ax[1].plot(data[:, 1], data[:, 2], color="blue", label="Proof Trace Retrieval\nMSE")
ax[1].set_yscale("log")
ax[1].set_ylim(0, np.max(data[:, 2]))
ax[1].set_xticks([])
ax[1].set_xlim(0, 1000)
ax[1].legend()


data = get_csv(filename="run_3Aautocorrelation-tag-trace_mse.csv")
ax[2].plot(data[:, 1], data[:, 2], color="blue", label="Autocorrelation Trace Retrieval\nMSE")
ax[2].set_yscale("log")
ax[2].set_ylim(0, np.max(data[:, 2]))
ax[2].set_xlim(0, 1000)
ax[2].set_xlabel("Epoch")
ax[2].legend()
fig.savefig("./3unsupervisedretrievalmethods1000.png")






plt.show()

