#!/bin/bash
name=JJJ_sample4_noise_resistant_network_various_noise_per_trace2
python supervised_retrieval.py $name
python supervised_retrieval_noise_test_results_plot.py $name

# name=JJJ_sample4_noise_resistant_network_various_noise_per_trace1
# python supervised_retrieval.py $name
# python supervised_retrieval_noise_test_results_plot.py $name


