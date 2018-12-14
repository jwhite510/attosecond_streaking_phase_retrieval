import numpy as np
import matplotlib.pyplot as plt
import pickle
import crab_tf2
import tensorflow as tf



def compare_traces(trace1, trace2):

    fig = plt.figure()
    gs = fig.add_gridspec(3,2)

    mse = (1 / len(trace1)) * np.sum(trace1 - trace2)**2

    ax = fig.add_subplot(gs[0, :])
    ax.pcolormesh(trace1.reshape(len(crab_tf2.p_values), len(crab_tf2.tau_values)), cmap='jet')

    ax = fig.add_subplot(gs[1, :])
    ax.pcolormesh(trace2.reshape(len(crab_tf2.p_values), len(crab_tf2.tau_values)), cmap='jet')
    ax.text(0.1, 0.92, 'mse: {}'.format(mse), transform=ax.transAxes, backgroundcolor='white')

    # plot the difference
    ax = fig.add_subplot(gs[2,:])
    diff = trace1 - trace2
    im = ax.pcolormesh(diff.reshape(len(crab_tf2.p_values), len(crab_tf2.tau_values)), cmap='jet')
    plt.colorbar(im, ax=ax)




def plot_trace(trace, xuv_field, ir_field, mse=None):


    fig = plt.figure()
    gs = fig.add_gridspec(3,2)

    ax = fig.add_subplot(gs[0,0])
    ax.plot(np.real(xuv_field), color='blue')
    ax.plot(np.imag(xuv_field), color='red')

    # get the signal in time
    out_xuv_time = sess.run(crab_tf2.xuv_time_domain, feed_dict={crab_tf2.xuv_cropped_f: xuv_field})
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(np.real(out_xuv_time), color='blue')


    # generate signal
    ax = fig.add_subplot(gs[1, :])
    image = sess.run(crab_tf2.image, feed_dict={crab_tf2.xuv_cropped_f: xuv_field, crab_tf2.ir_cropped_f: ir_field})
    ax.pcolormesh(image, cmap='jet')


    ax = fig.add_subplot(gs[2, :])
    ax.pcolormesh(trace.reshape(len(crab_tf2.p_values), len(crab_tf2.tau_values)), cmap='jet')


    if mse:
        ax.text(0.5, 0.1, str(mse), backgroundcolor='white', transform=ax.transAxes)


with open('ambiguity_pickles.p', 'rb') as file:
    ambiguity_pickles = pickle.load(file)


#print(ambiguity_pickles['similar_trace_pickles'])
#print(np.shape(ambiguity_pickles['similar_trace_pickles']))


#print(ambiguity_pickles['similar_trace_field_pickles'])
#print(np.shape(ambiguity_pickles['similar_trace_field_pickles']))

traces = ambiguity_pickles['similar_trace_pickles']
xuv_fields = ambiguity_pickles['similar_trace_xuv_field_pickles']
ir_fields = ambiguity_pickles['similar_trace_ir_field_pickles']


init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()


    index1 = 0
    index2 = 6

    plot_trace(trace=traces[index1], xuv_field=xuv_fields[index1], ir_field=ir_fields[index1])

    plot_trace(trace=traces[index2], xuv_field=xuv_fields[index2], ir_field=ir_fields[index2])

    compare_traces(trace1=traces[index1], trace2=traces[index2])


plt.show()