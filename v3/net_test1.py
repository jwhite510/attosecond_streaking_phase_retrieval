import tables
import supervised_retrieval
from xuv_spectrum import spectrum
import tensorflow as tf
import pickle
import numpy as np
import tf_functions
from phase_parameters import params
import measured_trace.get_trace as get_measured_trace
import generate_data3
import matplotlib.pyplot as plt
# import network3
import importlib


def get_closest_params(retrieved, network_in):
    network3 = importlib.import_module("models.network3_"+network_in)

    ir_values_in = tf.placeholder(tf.float32, shape=[None, 4])
    ir_label = network3.convert_ir_params(ir_values_in)

    with tf.Session() as sess:

        smallest_error_data = None
        smallest_error_index = None
        smallest_error_data_ls = []
        smallest_error_index_ls = []
        cost_lowest = 100
        getdata = network3.GetData(batch_size=10)

        while getdata.batch_index < getdata.samples:
            current_index = getdata.batch_index
            batch_x, batch_y = getdata.next_batch()

            print("getdata.batch_index =>", getdata.batch_index)
            for index, batch_y_i in enumerate(batch_y):
                # find the closest match
                ir_y = batch_y_i[params.xuv_phase_coefs:]
                xuv_y = batch_y_i[:params.xuv_phase_coefs]

                # convert ir params to get phase
                ir_label_out = sess.run(ir_label, feed_dict={ir_values_in:np.array([ir_y])})
                ir_label_out_ret = sess.run(ir_label, feed_dict={ir_values_in:retrieved["ir_params_pred"]})

                xuv_label = xuv_y[1:]
                xuv_label_ret = retrieved["xuv_retrieved"][0,1:]

                # calculate the smallest difference according to the cost function
                cost = np.mean((ir_label_out-ir_label_out_ret)**2) + np.mean((xuv_label-xuv_label_ret)**2)

                if cost < cost_lowest:
                    # print("new lowest cost:", cost)
                    # print("retrieved['predicted_coefficients_params'] =>")
                    # print(retrieved['predicted_coefficients_params'])
                    # print("batch_y_i =>")
                    # print(batch_y_i)
                    cost_lowest = cost
                    smallest_error_data = np.array(batch_y_i)
                    smallest_error_index = current_index + index
                    smallest_error_data_ls.append(smallest_error_data)
                    smallest_error_index_ls.append(smallest_error_index)

        obj = {
                "smallest_error_index":smallest_error_index,
                "smallest_error_data":smallest_error_data,
                "smallest_error_data_ls":smallest_error_data_ls,
                "smallest_error_index_ls":smallest_error_index_ls
                }

    return obj

def open_data_index(index, data_type=None):
    # open the trace corresponding to this error number
    if data_type is not None:
        if data_type=="train":
            hdf5_file = tables.open_file('train3.hdf5', mode="r")
        elif data_type=="test":
            hdf5_file = tables.open_file('test3.hdf5', mode="r")
        else:
            raise ValueError("not test or train")
    else:
        hdf5_file = tables.open_file('train3.hdf5', mode="r")

    xuv_coefs = hdf5_file.root.xuv_coefs[index:index + 1, :]
    ir_params = hdf5_file.root.ir_params[index:index + 1, :]
    appended_label_batch = np.append(xuv_coefs, ir_params, 1)
    trace_batch = hdf5_file.root.noise_trace[index:index + 1, :]
    hdf5_file.close()
    return trace_batch, appended_label_batch

if __name__ == "__main__":
    # retrieve measured trace
    measured_trace = get_measured_trace.trace
    supervised_retrieval_obj = supervised_retrieval.SupervisedRetrieval("EEFOV_increaseI_1")
    measured_retrieve_output = supervised_retrieval_obj.retrieve(measured_trace)

    # get the most similar trace from training data
    obj = get_closest_params(measured_retrieve_output, "EEFOV_increaseI_1")
    with open("closest_trace_training_data.p", "wb") as file:
        pickle.dump(obj, file)
    # open the closest trace
    with open("closest_trace_training_data.p", "rb") as file:
        obj = pickle.load(file)

    smallest_error_index = obj["smallest_error_index"]
    smallest_error_data = obj["smallest_error_data"]
    smallest_error_data_ls = obj["smallest_error_data_ls"]
    smallest_error_index_ls = obj["smallest_error_index_ls"]

    # get the reconstruction of the measured trace with known xuv coefficients
    with open("orignal_retrieved_xuv_coefs_newI.p", "rb") as file:
        obj = pickle.load(file)

    # take the error of the retrieved phase from the sample in the training data and the retrieved parameters from the initial retrieval

    # use this for retrieval
    trace, label = open_data_index(smallest_error_index+0)
    trace = trace.reshape(len(params.K), len(params.delay_values))
    plt.figure(1)
    plt.pcolormesh(trace)
    plt.savefig("index_p0")


    # use this for retrieval
    trace, label = open_data_index(smallest_error_index+1)
    trace = trace.reshape(len(params.K), len(params.delay_values))
    plt.figure(2)
    plt.pcolormesh(trace)
    plt.savefig("index_p1")


    # use this for retrieval
    trace, label = open_data_index(smallest_error_index+2)
    trace = trace.reshape(len(params.K), len(params.delay_values))
    plt.figure(3)
    plt.pcolormesh(trace)
    plt.savefig("index_p2")


    # use this for retrieval
    trace, label = open_data_index(smallest_error_index+3)
    trace = trace.reshape(len(params.K), len(params.delay_values))
    plt.figure(4)
    plt.pcolormesh(trace)
    plt.savefig("index_p3")


    # use this for retrieval
    trace, label = open_data_index(smallest_error_index+4)
    trace = trace.reshape(len(params.K), len(params.delay_values))
    plt.figure(5)
    plt.pcolormesh(trace)
    plt.savefig("index_p4")


    # use this for retrieval
    trace, label = open_data_index(smallest_error_index+5)
    trace = trace.reshape(len(params.K), len(params.delay_values))
    plt.figure(6)
    plt.pcolormesh(trace)
    plt.savefig("index_p5")


    # use this for retrieval
    trace, label = open_data_index(smallest_error_index+6)
    trace = trace.reshape(len(params.K), len(params.delay_values))
    plt.figure(7)
    plt.pcolormesh(trace)
    plt.savefig("index_p6")

    exit()

    measured_retrieve_output["trace_recons"]
    count_num = 50
    noise_trace_recons_added_noise = generate_data3.add_shot_noise(measured_retrieve_output["trace_recons"], count_num)
    retrieve_output = supervised_retrieval_obj.retrieve(noise_trace_recons_added_noise)

    retrieve_output = supervised_retrieval_obj.retrieve(trace)


    plt.figure(2)
    plt.pcolormesh(trace)

    import ipdb; ipdb.set_trace() # BREAKPOINT
    print("BREAKPOINT")


