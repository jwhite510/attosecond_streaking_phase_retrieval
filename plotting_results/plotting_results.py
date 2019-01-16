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
#filenames.append('./12_28_18_pictures/run_reg_conv_net_lr0001-tag-train_mse.csv')
#filenames.append('./12_28_18_pictures/run_reg_conv_net_ir_randomphase_lr0001-tag-train_mse.csv')
#filenames.append('./12_28_18_pictures/run_reg_conv_net_ir_randomphase_twodense_lr0001-tag-train_mse.csv')
#filenames.append('./12_28_18_pictures/run_reg_conv_net_ir_randomphase_onedense1024_lr0001-tag-train_mse.csv')
#filenames.append('./12_28_18_pictures/run_reg_conv_net_ir_randomphase_normalized_lr0001-tag-train_mse.csv')
#filenames.append('./gaussian_random_ir_phase_result/run_gaussian_ir_constantphase_lr0001-tag-train_mse.csv')
#filenames.append('./gaussian_random_ir_phase_result/run_gaussian_ir_randomphase_lr0001-tag-train_mse.csv')
#filenames.append('./gaussian_random_ir_phase_result/run_gaussian_ir_randomphase_lr0001_moresteps-tag-train_mse.csv')

# sample from 1_1_2019
filenames.append('./1_1_2019/run_gaussian_ir_randomrandomphase_lr0001-tag-test_mse.csv')
filenames.append('./1_1_2019/run_gaussian_ir_randomrandomphase_lr0001-tag-train_mse.csv')

filenames.append('./1_1_2019/run_gaussian_ir_randomphase_lr0001_moresteps-tag-test_mse.csv')
filenames.append('./1_1_2019/run_gaussian_ir_randomphase_lr0001_moresteps-tag-train_mse.csv')

filenames.append('./1_1_2019/run_gaussian_ir_randomphase_40ksamples_lr0001-tag-test_mse.csv')
filenames.append('./1_1_2019/run_gaussian_ir_randomphase_40ksamples_lr0001-tag-train_mse.csv')

filenames.append('./1_1_2019/run_gaussian_dtau130as_without_noise_constant_ir-tag-test_mse.csv')
filenames.append('./1_1_2019/run_gaussian_dtau130as_without_noise_constant_ir-tag-train_mse.csv')

filenames.append('./1_1_2019/run_gaussian_dtau130as_without_noise_constant_ir_lr0001-tag-test_mse.csv')
filenames.append('./1_1_2019/run_gaussian_dtau130as_without_noise_constant_ir_lr0001-tag-train_mse.csv')

filenames.append('./1_1_2019/run_gaussian_dtau130as_with_noise_constant_ir_lr0001-tag-test_mse.csv')
filenames.append('./1_1_2019/run_gaussian_dtau130as_with_noise_constant_ir_lr0001-tag-train_mse.csv')

filenames.append('./1_1_2019/run_gaussian_dtau130as_with_noise_constant_ir_lr0001_cubicphase_80ksamples-tag-train_mse.csv')
filenames.append('./1_1_2019/run_gaussian_dtau130as_with_noise_constant_ir_lr0001_cubicphase_80ksamples-tag-test_mse.csv')







colors = ["#0B464B", "#EA530E", "#401F3E", "#FF5964", "#083D77", "#151918", "#ED241A"]
colors1 = []
for color in colors:
    colors1.append(color)
    colors1.append(color)



fig, ax = plt.subplots(1, 1, figsize=(10,8))


#for i, filename, color in zip(enumerate(filenames), colors1):

for i, filename, color in zip(range(len(filenames)), filenames, colors1):

    step, mse = get_run(filename)

    if 'train' in filename:
        # plot with label
        run_name = filename.split('/')[2].split('-tag-')[0]
        ax.plot(step, mse, label=run_name, color=color)
    else:
        ax.plot(step, mse, linestyle=':', color=color, linewidth=3)


    ax.set_ylim(0, 0.004)
    ax.set_xlim(-20, 1100)
    ax.set_xlabel('Step')
    ax.set_ylabel('MSE')
    ax.legend()
    plt.savefig('./1_1_2019_plots/{}.png'.format(i))





plt.show()







