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

    def __init__(self, generations, pop_size, run_name, measured_trace):
        print("initialize")
        self.generations = generations
        self.pop_size = pop_size
        self.run_name = run_name
        self.measured_trace = measured_trace
        # self.plot_axes = create_exp_plot_axes()
        # self.tensorboard_tools = create_tensorboard_tools()
        # create tensorboard rmse measurer

        self.tf_generator_graphs = self.initialize_xuv_ir_trace_graphs()
        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter("./tensorboard_graph_ga/" + run_name)

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

        # fits = [ind.fitness.values[0] for ind in pop]

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
        rmse = self.calc_vecs_and_rmse(individual)

        return rmse

    def calc_vecs_and_rmse(self, individual, plot_and_graph=None):
        xuv_values = np.append([0], individual["xuv"])
        # append 0 for linear phase

        ir_values = individual["ir"]

        generated_trace = self.sess.run(self.tf_generator_graphs["image"],
                                        feed_dict={self.tf_generator_graphs["xuv_coefs_in"]: xuv_values.reshape(1, -1),
                                                   self.tf_generator_graphs["ir_values_in"]: ir_values.reshape(1, -1)})

        # calculate rmse
        trace_rmse = np.sqrt(
            (1 / len(self.measured_trace.reshape(-1))) * np.sum(
                (self.measured_trace - generated_trace.reshape(-1)) ** 2))

        if plot_and_graph:
            print("plot the trace")

            # add tensorboard value
            self.add_tensorboard_values(trace_rmse)


            # if "actual_fields" in plot_and_graph:
            #     # simulated trace
            #     predicted_fields = {}
            #     predicted_fields["ir_f"] = sess.run(tf_generator_graphs["ir_E_prop"]["f_cropped"],
            #                                         feed_dict={tf_generator_graphs["ir_values_in"]: ir_values.reshape(1, -1)})
            #     predicted_fields["xuv_f"] = sess.run(tf_generator_graphs["xuv_E_prop"]["f_cropped"],
            #                                          feed_dict={tf_generator_graphs["xuv_coefs_in"]: xuv_values.reshape(1, -1)})
            #     # plot
            #     plot_image_and_fields(plot_and_graph=plot_and_graph,
            #                           predicted_fields=predicted_fields,
            #                           predicted_streaking_trace=generated_trace,
            #                           actual_streaking_trace=measured_trace,
            #                           rmse=trace_rmse)
            # else:
            #     # experimental trace
            #     predicted_fields = {}
            #     predicted_fields["ir_f"] = sess.run(tf_generator_graphs["ir_E_prop"]["f_cropped"],
            #                                         feed_dict={
            #                                             tf_generator_graphs["ir_values_in"]: ir_values.reshape(1, -1)})
            #     predicted_fields["xuv_f"] = sess.run(tf_generator_graphs["xuv_E_prop"]["f_cropped"],
            #                                          feed_dict={
            #                                              tf_generator_graphs["xuv_coefs_in"]: xuv_values.reshape(1, -1)})
            #     predicted_fields["xuv_t"] = sess.run(tf_generator_graphs["xuv_E_prop"]["t"],
            #                                          feed_dict={
            #                                              tf_generator_graphs["xuv_coefs_in"]: xuv_values.reshape(1, -1)})

            #     plot_image_and_fields_exp(plot_and_graph=plot_and_graph,
            #                           predicted_fields=predicted_fields,
            #                           predicted_streaking_trace=generated_trace,
            #                           actual_streaking_trace=measured_trace,
            #                           rmse=trace_rmse)






        return trace_rmse

    def add_tensorboard_values(self, rmse):
        # summ = self.sess.run(self.tensorboard_tools["unsupervised_mse_tb"],
        #                 feed_dict={self.tensorboard_tools["rmse_tb"]: rmse})
        # self.writer.add_summary(summ, global_step=generation)
        # self.writer.flush()
        pass

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
            self.calc_vecs_and_rmse(best_ind, plot_and_graph=True)

        # return the rmse of final result
        best_ind = tools.selBest(self.pop, 1)[0]
        return self.calc_vecs_and_rmse(best_ind, plot_and_graph=True)

    def initialize_xuv_ir_trace_graphs(self):
        # initialize XUV generator
        xuv_phase_coeffs = phase_parameters.params.xuv_phase_coefs
        xuv_coefs_in = tf.placeholder(tf.float32, shape=[None, xuv_phase_coeffs])
        xuv_E_prop = tf_functions.xuv_taylor_to_E(xuv_coefs_in)


        # IR creation
        ir_values_in = tf.placeholder(tf.float32, shape=[None, 4])
        ir_E_prop = tf_functions.ir_from_params(ir_values_in)

        # initialize streaking trace generator
        # Neon


        # construct streaking image
        image = tf_functions.streaking_trace(xuv_cropped_f_in=xuv_E_prop["f_cropped"][0],
                                             ir_cropped_f_in=ir_E_prop["f_cropped"][0],
                                             )

        tf_graphs = {}
        tf_graphs["xuv_coefs_in"] = xuv_coefs_in
        tf_graphs["ir_values_in"] = ir_values_in
        tf_graphs["xuv_E_prop"] = xuv_E_prop
        tf_graphs["ir_E_prop"] = ir_E_prop
        tf_graphs["image"] = image

        return tf_graphs

    def create_population(self, create_individual, n):
        # return a list as the population
        population = []
        for i in range(n):
            population.append(create_individual())

        return population



def genetic_algorithm(generations, pop_size, run_name, tf_generator_graphs, measured_trace,
                      tensorboard_tools, plot_and_graph):

    with tf.Session() as sess:

        writer = tf.summary.FileWriter("./tensorboard_graph_ga/" + run_name)

        # minimize the fitness function (-1.0)
        creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
        # create individual
        creator.create("Individual", dict, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        # individual creator
        toolbox.register("create_individual", create_individual)

        toolbox.register("create_population", create_population, toolbox.create_individual)

        toolbox.register("evaluate", evaluate)

        toolbox.register("select", tools.selTournament, tournsize=4)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.5)

        # create the initial population
        pop = toolbox.create_population(n=pop_size)
        # print(pop[0].fitness.values)

        # evaluate and assign fitness numbers
        # fitnesses = list(map(toolbox.evaluate, pop))
        fitnesses = [toolbox.evaluate(p, measured_trace, tf_generator_graphs, sess) for p in pop]
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit,

        print("  Evaluated %i individuals" % len(pop))

        # fits = [ind.fitness.values[0] for ind in pop]

        # MUTPB is the probability for mutating an individual
        CXPB, MUTPB, MUTPB2 = 0.05, 0.05, 0.1
        # CXPB, MUTPB, MUTPB2 = 1.0, 1.0, 1.0

        # Variable keeping track of the number of generations
        g = 0

        while g <= generations:
            g = g + 1
            print("-- Generation %i --" % g)

            offspring = toolbox.select(pop, len(pop))

            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):

                # cross two individuals with probability CXPB
                re_evaluate = False
                for vector in ['xuv', 'ir']:
                    if random.random() < CXPB:
                        toolbox.mate(child1[vector], child2[vector])
                        re_evaluate = True
                if re_evaluate:
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                # mutate an individual with probability MUTPB
                re_evaluate = False
                for vector in ['xuv', 'ir']:
                    if random.random() < MUTPB:
                        toolbox.mutate(mutant[vector])
                        re_evaluate = True
                if re_evaluate:
                    del mutant.fitness.values

            for mutant in offspring:
                # mutate an individual with probabililty MUTPB2
                re_evaluate = False
                for vector in ['xuv', 'ir']:
                    if random.random() < MUTPB2:
                        # tools.mutGaussian(mutant[vector], mu=0.0, sigma=0.2, indpb=0.2)
                        tools.mutGaussian(mutant[vector], mu=0.0, sigma=0.1, indpb=0.6)
                        re_evaluate = True

                if re_evaluate:
                    del mutant.fitness.values


            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # fitnesses = [toolbox.evaluate(inv, input_data, frequency_space, spline_params, sess) for inv in invalid_ind]
            fitnesses = [toolbox.evaluate(inv, measured_trace, tf_generator_graphs, sess) for inv in invalid_ind]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit,

            print("  Evaluated %i individuals" % len(invalid_ind))

            # The population is entirely replaced by the offspring
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5

            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)

            print("-- End of (successful) evolution -- gen {}".format(str(g)))

            best_ind = tools.selBest(pop, 1)[0]
            # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

            # calc_vecs_and_rmse(best_ind, input_data, frequency_space, spline_params, sess, axes=axes, generation=g, writer=writer,
            #                    tensorboard_tools=tensorboard_tools, run_name=run_name)


            plot_and_graph["generation"] = g
            plot_and_graph["writer"] = writer
            plot_and_graph["tensorboard_tools"] = tensorboard_tools
            plot_and_graph["run_name"] = run_name
            calc_vecs_and_rmse(best_ind, measured_trace, tf_generator_graphs, sess, plot_and_graph=plot_and_graph)



        # return the rmse of final result
        best_ind = tools.selBest(pop, 1)[0]
        return calc_vecs_and_rmse(best_ind, measured_trace, tf_generator_graphs, sess, plot_and_graph=plot_and_graph)



if __name__ == "__main__":



    #..................................
    # .....retrieve measured trace.....
    #..................................
    # plot_axes = create_exp_plot_axes()
    # plot_and_graph = {}
    # plot_and_graph["plot_axes"] = plot_axes
    measured_trace = get_measured_trace.trace
    measured_trace = measured_trace.reshape(1, -1)

    genetic_algorithm = GeneticAlgorithm(generations=5, pop_size=5, 
                        run_name="experimental_retrieval1", measured_trace=measured_trace)
    
    genetic_algorithm.run()



    # genetic_algorithm(generations=500, pop_size=5000, run_name="experimental_retrieval1", tf_generator_graphs=tf_generator_graphs,
    #                   measured_trace=measured_trace, tensorboard_tools=tensorboard_tools, plot_and_graph=plot_and_graph)
