from crab_tf2 import *
import tables
import scipy.constants as sc


def plot_opened_file(xuv_t, ir_t, trace):


    fig = plt.figure()
    gs = fig.add_gridspec(2,2)

    # plot xuv
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(xuv.tmat, np.real(xuv_t), color='blue')

    # plot ir
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ir.tmat, np.real(ir_t), color='blue')

    # plot trace
    ax = fig.add_subplot(gs[1,:])
    ax.pcolormesh(tau_values, p_values, trace.reshape(len(p_values), len(tau_values)), cmap='jet')


index = 17

test_open_file = 'attstrace_train2_processed.hdf5'
# test_open_file = 'attstrace_test2_processed.hdf5'
with tables.open_file(test_open_file, mode='r') as processed_data:

    xuv_t1 = processed_data.root.xuv_t[index, :]
    ir_t1 = processed_data.root.ir_t[index, :]
    trace1 = processed_data.root.trace[index, :]

plot_opened_file(xuv_t1, ir_t1, trace1)



test_open_file = 'attstrace_train2.hdf5'
# test_open_file = 'attstrace_test2_processed.hdf5'
with tables.open_file(test_open_file, mode='r') as processed_data:

    xuv_t2 = processed_data.root.xuv_t[index, :]
    ir_t2 = processed_data.root.ir_t[index, :]
    trace2 = processed_data.root.trace[index, :]

plot_opened_file(xuv_t2, ir_t2, trace2)

# compare mse
fig = plt.figure()
gs = fig.add_gridspec(2,2)

ax = fig.add_subplot(gs[0,:])
ax.pcolormesh(trace2.reshape(len(p_values), len(tau_values)), cmap='jet')
ax.text(0.1, 0.8, "without noise", transform=ax.transAxes, backgroundcolor='white')


ax = fig.add_subplot(gs[1,:])
ax.pcolormesh(trace1.reshape(len(p_values), len(tau_values)), cmap='jet')
ax.text(0.1, 0.8, "with noise", transform=ax.transAxes, backgroundcolor='white')

# check the min and max values
print("min: ", np.min(trace1))
print("max: ", np.max(trace1))

print("min: ", np.min(trace2))
print("max: ", np.max(trace2))


# get the rmse
trace1reshaped = trace1.reshape(-1)
trace2reshaped = trace2.reshape(-1)
trace_rmse = np.sqrt((1 / len(trace2reshaped)) * np.sum((trace1reshaped - trace2reshaped) ** 2))
print("trace_rmse: ",trace_rmse)

# plot the rmse
ax.text(0.1, 0.65, "rmse: {}".format(str(round(trace_rmse, 5))), transform=ax.transAxes, backgroundcolor='white')

plt.savefig("./rmse_noise_check/{}.png".format(str(index)))

plt.show()


