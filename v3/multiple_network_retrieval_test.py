import supervised_retrieval
import pickle
import numpy as np
from phase_parameters import params
import measured_trace.get_trace as get_measured_trace
import generate_data3
import matplotlib.pyplot as plt



def run_retrievals_on_networks():
    # use one of the networks to retrieve the measured trace
    measured_trace = get_measured_trace.trace
    supervised_retrieval_obj = supervised_retrieval.SupervisedRetrieval("MLMRL_noise_resistant_net_angle_18")
    retrieve_output = supervised_retrieval_obj.retrieve(measured_trace)

    # get the reconstruction of the measured trace with known xuv xuv coefficients
    reconstructed_trace = retrieve_output["trace_recons"]
    orignal_retrieved_xuv_coefs = retrieve_output["xuv_retrieved"]

    # add noise to reconstructed trace
    count_num = 50
    noise_trace_recons_added_noise = generate_data3.add_shot_noise(reconstructed_trace, count_num)

    # delete the tensorflow graph
    del supervised_retrieval_obj

    # input the reconstructed trace to all the networks and see the variation in the output
    retrieved_xuv_cl = []
    for tf_model in [ "MLMRL_noise_resistant_net_angle_1",
                    "MLMRL_noise_resistant_net_angle_2",
                    "MLMRL_noise_resistant_net_angle_3",
                    "MLMRL_noise_resistant_net_angle_4",
                    "MLMRL_noise_resistant_net_angle_5",
                    "MLMRL_noise_resistant_net_angle_6",
                    "MLMRL_noise_resistant_net_angle_7",
                    "MLMRL_noise_resistant_net_angle_8",
                    "MLMRL_noise_resistant_net_angle_9",
                    "MLMRL_noise_resistant_net_angle_10",
                    "MLMRL_noise_resistant_net_angle_11",
                    "MLMRL_noise_resistant_net_angle_12",
                    "MLMRL_noise_resistant_net_angle_13",
                    "MLMRL_noise_resistant_net_angle_14",
                    "MLMRL_noise_resistant_net_angle_15",
                    "MLMRL_noise_resistant_net_angle_16",
                    "MLMRL_noise_resistant_net_angle_17",
                    "MLMRL_noise_resistant_net_angle_18"
                    ]:
        supervised_retrieval_obj = supervised_retrieval.SupervisedRetrieval(tf_model)
        retrieve_output = supervised_retrieval_obj.retrieve(noise_trace_recons_added_noise)
        retrieved_xuv_coefs = retrieve_output["xuv_retrieved"]
        del supervised_retrieval_obj

        # add the retrieved xuv coefs to list
        retrieved_xuv_cl.append(retrieved_xuv_coefs)

    return retrieved_xuv_cl, noise_trace_recons_added_noise



if __name__ == "__main__":
    """
    this .py file is for taking many trained networks and retrieving the measured trace and a
    reconstruction of the measured trace (then having known xuv coefficients) and performing
    multiple retrievals with many trained networks to look at the variation in retrieval
    """

    retrieved_xuv_cl, noise_trace_recons_added_noise = run_retrievals_on_networks()

    data = {}
    data["retrieved_xuv_cl"] = retrieved_xuv_cl
    data["noise_trace_recons_added_noise"] = noise_trace_recons_added_noise
    with open("multiple_net_retrieval_test.p", "wb") as file:
        pickle.dump(data, file)




