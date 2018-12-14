import numpy as np
import crab_tf2
import matplotlib.pyplot as plt
import tables
import pickle



class GetTraces():

    def __init__(self, filename):

        self.filename = filename


    def get_trace(self, index):

        with tables.open_file(self.filename, mode='r') as hdf5_file:
            trace = hdf5_file.root.trace[index, :]
            xuv = hdf5_file.root.xuv_f[index, :]
            ir = hdf5_file.root.ir_f[index, :]

        return trace, ir, xuv


def plot_trace(trace, ir_field, xuv_field, mse=None):


    fig = plt.figure()
    gs = fig.add_gridspec(2,3)

    ax = fig.add_subplot(gs[0,:])
    ax.plot(np.real(xuv_field), color='blue')
    ax.plot(np.imag(xuv_field), color='red')

    ax = fig.add_subplot(gs[1, :])
    ax.pcolormesh(trace.reshape(len(crab_tf2.p_values), len(crab_tf2.tau_values)), cmap='jet')


    if mse:
        ax.text(0.5, 0.1, str(mse), backgroundcolor='white', transform=ax.transAxes)





traces = GetTraces(filename='attstrace_train2_processed.hdf5')


original_trace, original_ir_field, original_xuv_field  = traces.get_trace(index=2)

plot_trace(original_trace, original_ir_field, original_xuv_field)


ambiguity_pickles = {}
ambiguity_pickles['similar_trace_pickles'] = []
ambiguity_pickles['similar_trace_ir_field_pickles']= []
ambiguity_pickles['similar_trace_xuv_field_pickles']= []
ambiguity_pickles['similar_trace_pickles'].append(original_trace)
ambiguity_pickles['similar_trace_xuv_field_pickles'].append(original_xuv_field)
ambiguity_pickles['similar_trace_ir_field_pickles'].append(original_ir_field)


found = 0
for i in range(120000):

    if i % 1000 == 0:
        print("searching index {}".format(i))

    # retrieve the trace and field
    trace, ir_field, xuv_field = traces.get_trace(index=i)

    # check the MSE
    mse = (1 / len(original_trace)) * np.sum(original_trace - trace) ** 2


    if mse < 0.000001:
    #if mse < 10000:
        found += 1
        print('found {}'.format(found))
        ambiguity_pickles['similar_trace_pickles'].append(trace)
        ambiguity_pickles['similar_trace_xuv_field_pickles'].append(xuv_field)
        ambiguity_pickles['similar_trace_ir_field_pickles'].append(ir_field)
        plot_trace(trace, ir_field, xuv_field, mse=mse)

with open('ambiguity_pickles.p', 'wb') as file:
    pickle.dump(ambiguity_pickles, file)

plt.show()








