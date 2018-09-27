import numpy as np
import matplotlib.pyplot as plt
import tables
import pickle
import generate_proof_traces
# open the pickles
try:
    with open('crab_tf_items.p', 'rb') as file:
        crab_tf_items = pickle.load(file)

    items = crab_tf_items['items']
    xuv_int_t = crab_tf_items['xuv_int_t']
    tmax = crab_tf_items['tmax']
    N = crab_tf_items['N']
    dt = crab_tf_items['dt']
    tauvec = crab_tf_items['tauvec']
    p_vec = crab_tf_items['p_vec']
    f0_ir = crab_tf_items['irf0']
    irEt = crab_tf_items['irEt']
    irtmat = crab_tf_items['irtmat']
    xuv_f0 = crab_tf_items['xuvf0']

except Exception as e:
    print(e)
    print('run crab_tf.py first to pickle the needed files')
    exit(0)


hdf5_file = tables.open_file("processed.hdf5", mode="r")


index_compare = 2
attstrace_compare = hdf5_file.root.attstrace[index_compare, :]
proof_compare = hdf5_file.root.proof[index_compare, :]
xuv_compare = hdf5_file.root.xuv_envelope[index_compare, :]

similar_attstraces = []
mses = []
threshold = 0.0001
maxsearchrange = 10000
for index_search in range(maxsearchrange):

    attstrace_search = hdf5_file.root.attstrace[index_search, :]
    proof_search = hdf5_file.root.proof[index_search, :]
    xuv_search = hdf5_file.root.xuv_envelope[index_search, :]

    print('Found {}, searching {} / {}'.format(len(mses), index_search, maxsearchrange))

    # compare
    mse = (1/(len(attstrace_search))) * np.sum((attstrace_search - attstrace_compare)**2)
    if mse < threshold:
        similar_attstraces.append(index_search)
        mses.append(mse)

print('found {} similar traces'.format(len(similar_attstraces)))
print(similar_attstraces)

print('plot '+str(len(similar_attstraces))+' traces?')
# answer = input()

# if answer =='y':
if True:


    for similar_index, mse in zip(similar_attstraces, mses):

        fig = plt.figure(constrained_layout=False, figsize=(7, 7))
        gs = fig.add_gridspec(2, 2)
        ax = fig.add_subplot(gs[0, 0])
        ax.pcolormesh(hdf5_file.root.attstrace[similar_index, :].reshape(len(p_vec),
                                                                                              len(tauvec)), cmap='jet')
        ax.text(0, 1, 'MSE:{}'.format(mse), backgroundcolor='white', transform=ax.transAxes)

        ax = fig.add_subplot(gs[1, 0])

        t = np.linspace(xuv_int_t[0], xuv_int_t[-1], generate_proof_traces.xuv_field_length)
        ax.plot(t, np.real(hdf5_file.root.xuv_envelope[similar_index, :]), color='blue')
        ax.plot(t, np.imag(hdf5_file.root.xuv_envelope[similar_index, :]), color='red')
        ax.plot([t[0], t[-1]], [0, 0], color='black', alpha=0.5)
        ax.plot([0, 0], [np.max(np.abs(xuv_compare)), -np.max(np.abs(xuv_compare))], color='black', alpha=0.5)


        ax = fig.add_subplot(gs[0, 1])
        ax.pcolormesh(attstrace_compare.reshape(len(p_vec),len(tauvec)), cmap='jet')

        ax = fig.add_subplot(gs[1, 1])
        ax.plot(t, np.real(xuv_compare), color='blue')
        ax.plot(t, np.imag(xuv_compare), color='red')
        ax.plot([t[0], t[-1]], [0, 0], color='black', alpha=0.5)
        ax.plot([0, 0], [np.max(np.abs(xuv_compare)), -np.max(np.abs(xuv_compare))], color='black', alpha=0.5)

    plt.show()
        # .reshape(len(p_vec), len(tauvec))


hdf5_file.close()
























