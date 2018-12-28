import numpy as np
import csv
import matplotlib.pyplot as plt



def get_run(filename):
    #with open('./12_28_18_pictures/' + test_name) as file:
    with open(filename) as file:
        reader = csv.reader(file)
        step, mse = [], []
        for row in np.array(list(reader))[1:]:
            step.append(float(row[1]))
            mse.append(float(row[2]))
        mse = np.array(mse)
        step = np.array(step)

    return step, mse



filenames = []
filenames.append('./12_28_18_pictures/run_reg_conv_net_lr0001-tag-train_mse.csv')
filenames.append('./12_28_18_pictures/run_reg_conv_net_ir_randomphase_lr0001-tag-train_mse.csv')
filenames.append('./12_28_18_pictures/run_reg_conv_net_ir_randomphase_twodense_lr0001-tag-train_mse.csv')
filenames.append('./12_28_18_pictures/run_reg_conv_net_ir_randomphase_onedense1024_lr0001-tag-train_mse.csv')
filenames.append('./12_28_18_pictures/run_reg_conv_net_ir_randomphase_normalized_lr0001-tag-train_mse.csv')

fig, ax = plt.subplots(1, 1, figsize=(10,5))

for i, filename in enumerate(filenames):

    step, mse = get_run(filename)
    run_name = filename.split('/')[2].split('-tag-')[0]
    ax.plot(step, mse, label=run_name)
    ax.set_ylim(0, 0.03)
    ax.set_xlabel('Step')
    ax.set_ylabel('MSE')
    ax.legend()
    plt.savefig('./{}.png'.format(i))





plt.show()







