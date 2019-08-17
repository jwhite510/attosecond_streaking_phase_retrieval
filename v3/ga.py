import numpy as np
import matplotlib.pyplot as plt
import random
from deap import base
from deap import creator
from deap import tools
import os
import tensorflow as tf
from scipy.interpolate import BSpline
import pickle
import datetime
import network3
import tables
from ir_spectrum import ir_spectrum
from xuv_spectrum import spectrum
import unsupervised_retrieval
import tf_functions
import phase_parameters.params
import measured_trace.get_trace as get_measured_trace


class GeneticAlgorithm():

    def __init__(self, generations, pop_size, run_name, measured_trace, retrieval, bootstrap=False):
        """
        bootstrap: dictionary
        bootstrap["indexes"] : a numpy array of the index values for bootstrap method (2/3 length of trace)
                                    shape (-1)
        
        """
        print("initialize")
        self.bootstrap = bootstrap
        self.method = "Genetic Algorithm"
        self.retrieval = retrieval
        self.generations = generations
        self.pop_size = pop_size
        self.run_name = run_name
        self.measured_trace = measured_trace
        # self.plot_axes = create_exp_plot_axes()
        self.tf_graphs = self.initialize_xuv_ir_trace_graphs()
        # create tensorboard mse measurer
        self.writer = tf.summary.FileWriter("./tensorboard_graph_ga/" + run_name)

        if self.retrieval == "normal":
            self.trace_mse_tb = tf.summary.scalar("trace_mse", self.tf_graphs["error"]["trace_mse"])
        elif self.retrieval == "proof":
            self.trace_mse_tb = tf.summary.scalar("trace_mse", self.tf_graphs["error"]["proof_mse"])
        elif self.retrieval == "autocorrelation":
            self.trace_mse_tb = tf.summary.scalar("trace_mse", self.tf_graphs["error"]["autocorr_mse"])

        self.sess = tf.Session()

        # create plot axes, share from unsupervised learning plotting
        self.axes = unsupervised_retrieval.create_plot_axes()

        # minimize the fitness function (-1.0)
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
        # create individual
        creator.create("Individual", dict, fitness=creator.FitnessMax)
        self.toolbox = base.Toolbox()
        # individual creator
        self.toolbox.register("create_individual", self.create_individual)
        self.toolbox.register("create_population", self.create_population, self.toolbox.create_individual)
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=4)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)
        # create the initial population
        self.pop = self.toolbox.create_population(n=pop_size)
        # evaluate and assign fitness numbers
        fitnesses = [self.toolbox.evaluate(p) for p in self.pop]
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit,

        print("  Evaluated %i individuals" % len(self.pop))

        # MUTPB is the probability for mutating an individual
        self.CXPB, self.MUTPB, self.MUTPB2 = 0.05, 0.05, 0.1
        # self.CXPB, self.MUTPB, self.MUTPB2 = 1.0, 1.0, 1.0

        # Variable keeping track of the number of generations
        self.g = 0

    def create_individual(self):
        individual = creator.Individual()

        # random numbers betwwen -1 and 1
        xuv_values = (2 * np.random.rand(4) - 1.0)
        individual["xuv"] = xuv_values

        # random numbers betwwen -1 and 1
        ir_values = (2 * np.random.rand(4) - 1.0)
        individual["ir"] = ir_values

        return individual

    def evaluate(self, individual):
        mse = self.calc_vecs_and_mse(individual)

        return mse

    def get_trace_and_rmse(self, individual):
        mse = self.calc_vecs_and_mse(individual)
        
        xuv_values = np.append([0], individual["xuv"])
        # append 0 for linear phase
        ir_values = individual["ir"]

        feed_dict = {self.tf_graphs["xuv_coefs_in"]: xuv_values.reshape(1, -1),
                     self.tf_graphs["ir_values_in"]: ir_values.reshape(1, -1)}
        
        if self.retrieval == "normal":
            trace = self.sess.run(self.tf_graphs["reconstructed"]["trace"],
                                                   feed_dict=feed_dict)

        elif self.retrieval == "proof":
            trace = self.sess.run(self.tf_graphs["reconstructed"]["proof"],
                                                   feed_dict=feed_dict)

        elif self.retrieval == "autocorrelation":
            trace = self.sess.run(self.tf_graphs["reconstructed"]["autocorrelation"],
                                                   feed_dict=feed_dict)

        return trace, mse

    def calc_vecs_and_mse(self, individual, plot_and_graph=None):
        xuv_values = np.append([0], individual["xuv"])
        # append 0 for linear phase
        ir_values = individual["ir"]

        feed_dict = {self.tf_graphs["xuv_coefs_in"]: xuv_values.reshape(1, -1),
                     self.tf_graphs["ir_values_in"]: ir_values.reshape(1, -1)}

        if self.retrieval == "normal":
            if self.bootstrap == False:
                # calculate mse for normal trace
                trace_mse = self.sess.run(self.tf_graphs["error"]["trace_mse"], feed_dict=feed_dict)
            else:
                # evaluate with bootstrap
                print("using bootstrap!")
                # add bootstrap indexes to the feed dict
                feed_dict[self.tf_graphs["error"]["bootstrap"]["normal"]["index_ph"]] = self.bootstrap["indexes"]
                trace_mse = self.sess.run(self.tf_graphs["error"]["bootstrap"]["normal"]["mse"], feed_dict=feed_dict)

        elif self.retrieval == "proof":
            if self.bootstrap == False:
                # calculate mse for proof trace
                trace_mse = self.sess.run(self.tf_graphs["error"]["proof_mse"], feed_dict=feed_dict)
            else:
                # evaluate with bootstrap
                print("using bootstrap!")
                # add bootstrap indexes to the feed dict
                feed_dict[self.tf_graphs["error"]["bootstrap"]["proof"]["index_ph"]] = self.bootstrap["indexes"]
                trace_mse = self.sess.run(self.tf_graphs["error"]["bootstrap"]["proof"]["mse"], feed_dict=feed_dict)

        elif self.retrieval == "autocorrelation":
            if self.bootstrap == False:
                # calculate mse for autocorrelation trace
                trace_mse = self.sess.run(self.tf_graphs["error"]["autocorr_mse"], feed_dict=feed_dict)
            else:
                # evaluate with bootstrap
                print("using bootstrap!")
                # add bootstrap indexes to the feed dict
                feed_dict[self.tf_graphs["error"]["bootstrap"]["auto"]["index_ph"]] = self.bootstrap["indexes"]
                trace_mse = self.sess.run(self.tf_graphs["error"]["bootstrap"]["auto"]["mse"], feed_dict=feed_dict)

        else:
            raise ValueError("retrieval must be either 'normal', 'proof', or 'autocorrelation'")

        if plot_and_graph:
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # ++++++++++calculate input and reconstructed traces++++++++++
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            input_traces = dict()
            input_traces["trace"] = self.sess.run(self.tf_graphs["measured"]["trace"])
            input_traces["proof"] = self.sess.run(self.tf_graphs["measured"]["proof"],
                                                   feed_dict=feed_dict)
            input_traces["autocorrelation"] = self.sess.run(self.tf_graphs["measured"]["autocorrelation"],
                                                   feed_dict=feed_dict)
            recons_traces = dict()
            recons_traces["trace"] = self.sess.run(self.tf_graphs["reconstructed"]["trace"],
                                                   feed_dict=feed_dict)
            recons_traces["proof"] = self.sess.run(self.tf_graphs["reconstructed"]["proof"],
                                                   feed_dict=feed_dict)
            recons_traces["autocorrelation"] = self.sess.run(self.tf_graphs["reconstructed"]["autocorrelation"],
                                                   feed_dict=feed_dict)

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # ++++++++++++++++++++calculate fields++++++++++++++++++++++
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            xuv_f = self.sess.run(self.tf_graphs["xuv_E_prop"]["f_cropped"], feed_dict=feed_dict)[0]
            xuv_f_phase = self.sess.run(self.tf_graphs["xuv_E_prop"]["phasecurve_cropped"], feed_dict=feed_dict)[0]
            xuv_f_full = self.sess.run(self.tf_graphs["xuv_E_prop"]["f"], feed_dict=feed_dict)[0]
            xuv_t = self.sess.run(self.tf_graphs["xuv_E_prop"]["t"], feed_dict=feed_dict)[0]
            ir_f = self.sess.run(self.tf_graphs["ir_E_prop"]["f_cropped"], feed_dict=feed_dict)[0]

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # ++++++++++++++++++plot fields and traces++++++++++++++++++++
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if self.retrieval == "normal":
                unsupervised_retrieval.plot_images_fields(axes=self.axes, traces_meas=input_traces,
                                   traces_reconstructed=recons_traces,
                                   xuv_f=xuv_f, xuv_f_phase=xuv_f_phase, xuv_f_full=xuv_f_full, xuv_t=xuv_t, ir_f=ir_f, i=self.g,
                                   run_name=self.run_name, true_fields=False, cost_function="trace",
                                   method=self.method, save_data_objs=True)
                plt.pause(0.00001)

            elif self.retrieval == "proof":
                unsupervised_retrieval.plot_images_fields(axes=self.axes, traces_meas=input_traces,
                                   traces_reconstructed=recons_traces,
                                   xuv_f=xuv_f, xuv_f_phase=xuv_f_phase, xuv_f_full=xuv_f_full, xuv_t=xuv_t, ir_f=ir_f, i=self.g,
                                   run_name=self.run_name, true_fields=False, cost_function="proof",
                                   method=self.method, save_data_objs=True)
                plt.pause(0.00001)

            elif self.retrieval == "autocorrelation":
                unsupervised_retrieval.plot_images_fields(axes=self.axes, traces_meas=input_traces,
                                   traces_reconstructed=recons_traces,
                                   xuv_f=xuv_f, xuv_f_phase=xuv_f_phase, xuv_f_full=xuv_f_full, xuv_t=xuv_t, ir_f=ir_f, i=self.g,
                                   run_name=self.run_name, true_fields=False, cost_function="autocorrelation",
                                   method=self.method, save_data_objs=True)
                plt.pause(0.00001)

            # add tensorboard value
            summ = self.sess.run(self.trace_mse_tb, feed_dict=feed_dict)
            self.writer.add_summary(summ, global_step=self.g)
            self.writer.flush()

        return trace_mse

    def run(self):
        print("run")
        while self.g <= self.generations:
            self.g = self.g + 1
            print("-- Generation %i --" % self.g)

            offspring = self.toolbox.select(self.pop, len(self.pop))

            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability CXPB
                re_evaluate = False
                for vector in ['xuv', 'ir']:
                    if random.random() < self.CXPB:
                        self.toolbox.mate(child1[vector], child2[vector])
                        re_evaluate = True
                if re_evaluate:
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                # mutate an individual with probability MUTPB
                re_evaluate = False
                for vector in ['xuv', 'ir']:
                    if random.random() < self.MUTPB:
                        self.toolbox.mutate(mutant[vector])
                        re_evaluate = True
                if re_evaluate:
                    del mutant.fitness.values

            for mutant in offspring:
                # mutate an individual with probabililty MUTPB2
                re_evaluate = False
                for vector in ['xuv', 'ir']:
                    if random.random() < self.MUTPB2:
                        # tools.mutGaussian(mutant[vector], mu=0.0, sigma=0.2, indpb=0.2)
                        tools.mutGaussian(mutant[vector], mu=0.0, sigma=0.1, indpb=0.6)
                        re_evaluate = True

                if re_evaluate:
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [self.toolbox.evaluate(inv) for inv in invalid_ind]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit,

            print("  Evaluated %i individuals" % len(invalid_ind))
            # The population is entirely replaced by the offspring
            self.pop[:] = offspring
            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in self.pop]

            length = len(self.pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
            print("-- End of (successful) evolution -- gen {}".format(str(self.g)))

            best_ind = tools.selBest(self.pop, 1)[0]

            # plot the best individual
            self.calc_vecs_and_mse(best_ind, plot_and_graph=True)

        # return the mse of final result
        best_ind = tools.selBest(self.pop, 1)[0]
        # return self.calc_vecs_and_mse(best_ind, plot_and_graph=True)
        phase_retrieved = self.get_phase_curve(best_ind)
        # trace and trace mse
        recons_trace, trace_mse = self.get_trace_and_rmse(best_ind)
        
        result = dict()
        result["field"] = phase_retrieved
        result["trace"] = dict()
        result["trace"]["reconstructed"] = recons_trace
        result["trace"]["mse"] = trace_mse
        return result

    def initialize_xuv_ir_trace_graphs(self):
        # calcualte autocorrelate and proof trace from measured trace
        tf_measured_trace = tf.constant(self.measured_trace, dtype=tf.float32)
        measured_auto_trace = tf_functions.autocorrelate(tf_measured_trace)
        measured_proof_trace = tf_functions.proof_trace(tf_measured_trace)["proof"]

        # initialize XUV generator
        xuv_phase_coeffs = phase_parameters.params.xuv_phase_coefs
        xuv_coefs_in = tf.placeholder(tf.float32, shape=[None, xuv_phase_coeffs])
        xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_in)

        # IR creation
        ir_values_in = tf.placeholder(tf.float32, shape=[None, 4])
        ir_E_prop = tf_functions.ir_from_params(ir_values_in)["E_prop"]

        # construct streaking image
        image = tf_functions.streaking_trace(xuv_cropped_f_in=xuv_E_prop["f_cropped"][0],
                                             ir_cropped_f_in=ir_E_prop["f_cropped"][0])
        # construct proof trace
        proof_recons = tf_functions.proof_trace(image)["proof"]
        auto_trace_recons = tf_functions.autocorrelate(image)

        # ++++++++++++++++++++++++++++++++++++++++
        # ++++++++++define loss functions+++++++++
        # ++++++++++++++++++++++++++++++++++++++++

        # define measured trace as constant to calculate MSE between regular trace
        measured_trace_flat_tens = tf.constant(self.measured_trace.reshape(1, -1), dtype=tf.float32)

        # normal trace cost function
        trace_mse = tf.losses.mean_squared_error(labels=measured_trace_flat_tens, predictions=tf.reshape(image, [1, -1]))

        # mean squared error for Autocorrelation
        autocorr_mse = tf.losses.mean_squared_error(labels=tf.reshape(measured_auto_trace, [1, -1]),
                                                    predictions=tf.reshape(auto_trace_recons, [1, -1]))

        # mean squared error for PROOF
        proof_mse = tf.losses.mean_squared_error(labels=tf.reshape(measured_proof_trace, [1, -1]),
                                                    predictions=tf.reshape(proof_recons, [1, -1]))

        # define bootstrap errors
        trace_boot_mse, trace_boot_ph = network3.calc_bootstrap_error(image, measured_trace_flat_tens)
        auto_boot_mse, auto_boot_ph = network3.calc_bootstrap_error(auto_trace_recons,  measured_auto_trace)
        proof_boot_mse, proof_boot_ph = network3.calc_bootstrap_error(proof_recons, measured_proof_trace)

        tf_graphs = dict()
        tf_graphs["measured"] = dict()
        tf_graphs["reconstructed"] = dict()
        tf_graphs["error"] = dict()

        tf_graphs["measured"]["trace"] = tf_measured_trace
        tf_graphs["measured"]["proof"] = measured_proof_trace
        tf_graphs["measured"]["autocorrelation"] = measured_auto_trace

        tf_graphs["reconstructed"]["trace"] = image
        tf_graphs["reconstructed"]["autocorrelation"] = auto_trace_recons
        tf_graphs["reconstructed"]["proof"] = proof_recons

        tf_graphs["error"]["trace_mse"] = trace_mse
        tf_graphs["error"]["autocorr_mse"] = autocorr_mse
        tf_graphs["error"]["proof_mse"] = proof_mse

        tf_graphs["xuv_coefs_in"] = xuv_coefs_in
        tf_graphs["ir_values_in"] = ir_values_in
        tf_graphs["xuv_E_prop"] = xuv_E_prop
        tf_graphs["ir_E_prop"] = ir_E_prop

        # bootstrap error
        tf_graphs["error"]["bootstrap"] = {}

        tf_graphs["error"]["bootstrap"]["normal"] = {}
        tf_graphs["error"]["bootstrap"]["normal"]["mse"] = trace_boot_mse
        tf_graphs["error"]["bootstrap"]["normal"]["index_ph"] = trace_boot_ph

        tf_graphs["error"]["bootstrap"]["auto"] = {}
        tf_graphs["error"]["bootstrap"]["auto"]["mse"] = auto_boot_mse
        tf_graphs["error"]["bootstrap"]["auto"]["index_ph"] = auto_boot_ph

        tf_graphs["error"]["bootstrap"]["proof"] = {}
        tf_graphs["error"]["bootstrap"]["proof"]["mse"] = proof_boot_mse
        tf_graphs["error"]["bootstrap"]["proof"]["index_ph"] = proof_boot_ph

        return tf_graphs

    def get_phase_curve(self, individual):
        xuv_values = np.append([0], individual["xuv"])
        # append 0 for linear phase
        ir_values = individual["ir"]
        feed_dict = {self.tf_graphs["xuv_coefs_in"]: xuv_values.reshape(1, -1),
                     self.tf_graphs["ir_values_in"]: ir_values.reshape(1, -1)}

        phase_curve = dict()
        phase_curve["cropped_phase"] = self.sess.run(self.tf_graphs["xuv_E_prop"]["phasecurve_cropped"], feed_dict=feed_dict)[0]
        phase_curve["f_full"] = self.sess.run(self.tf_graphs["xuv_E_prop"]["f"], feed_dict=feed_dict)[0]
        return phase_curve

    def create_population(self, create_individual, n):
        # return a list as the population
        population = []
        for i in range(n):
            population.append(create_individual())

        return population


if __name__ == "__main__":
    #..................................
    # .....retrieve measured trace.....
    #..................................
    # plot_axes = create_exp_plot_axes()
    # plot_and_graph = {}
    # plot_and_graph["plot_axes"] = plot_axes
    measured_trace = get_measured_trace.trace

    genetic_algorithm = GeneticAlgorithm(generations=300, pop_size=5000,
                        run_name="sample4_ga_1_normal", measured_trace=measured_trace, retrieval="normal")
    
    genetic_algorithm.run()



    # genetic_algorithm(generations=500, pop_size=5000, run_name="experimental_retrieval1", tf_generator_graphs=tf_generator_graphs,
    #                   measured_trace=measured_trace, tensorboard_tools=tensorboard_tools, plot_and_graph=plot_and_graph)
