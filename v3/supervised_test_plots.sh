#!/bin/bash
name=MLMN_noise_resistant_net_angle_1
python supervised_retrieval.py $name && python supervised_retrieval_noise_test_results_plot.py $name

name=MLMN_noise_resistant_net_angle_2
python supervised_retrieval.py $name && python supervised_retrieval_noise_test_results_plot.py $name

name=MLMN_noise_resistant_net_angle_3
python supervised_retrieval.py $name && python supervised_retrieval_noise_test_results_plot.py $name


