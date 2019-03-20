import csv
import numpy as np
import matplotlib.pyplot as plt



def get_csv(filename):
    with open(filename) as file:
        reader = csv.reader(file)
        content = np.array(list(reader))
        data = content[1:].astype(np.float)
    return data






data = get_csv(filename="run_test1_phasecurve-tag-train_mse_coef_params.csv")
plt.figure(1)
plt.plot(data[:, 1], data[:, 2], color="blue")
plt.plot([15, 15], [0, np.max(data[:, 2])], color="red")
plt.plot([30, 30], [0, np.max(data[:, 2])], color="red")
plt.ylabel("MSE")
plt.xlabel("Epoch")
plt.gca().set_yscale("log")
# plt.xlim(0, 60)
plt.title("Coefficient Parameters MSE")
plt.savefig("CoefficientParametersMSE_full.png")


data = get_csv(filename="run_test1_phasecurve-tag-train_mse_fields.csv")
plt.figure(2)
plt.plot(data[:, 1], data[:, 2], color="blue")
plt.plot([15, 15], [0, np.max(data[:, 2])], color="red")
plt.plot([30, 30], [0, np.max(data[:, 2])], color="red")
plt.ylabel("MSE")
plt.xlabel("Epoch")
plt.gca().set_yscale("log")
# plt.xlim(0, 60)
plt.title("Fields vector MSE")
plt.savefig("FieldsvectorMSE_full.png")


data = get_csv(filename="run_test1_phasecurve-tag-train_mse_phasecurve.csv")
plt.figure(3)
plt.plot(data[:, 1], data[:, 2], color="blue")
plt.plot([15, 15], [0, np.max(data[:, 2])], color="red")
plt.plot([30, 30], [0, np.max(data[:, 2])], color="red")
plt.ylabel("MSE")
plt.xlabel("Epoch")
plt.gca().set_yscale("log")
# plt.xlim(0, 60)
plt.title("phasecurve vector MSE")
plt.savefig("phasecurvevectorMSE_full.png")
plt.show()





