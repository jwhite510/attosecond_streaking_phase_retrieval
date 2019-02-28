import numpy as np
import matplotlib.pyplot as plt
import random
from deap import base
from deap import creator
from deap import tools
import os
import tensorflow as tf
# import unsupervised
# import crab_tf2
from scipy.interpolate import BSpline
import pickle
import datetime
import network3
import tables




def add_tensorboard_values(rmse, generation, sess, writer, tensorboard_tools):

    summ = sess.run(tensorboard_tools["unsupervised_mse_tb"],
                    feed_dict={tensorboard_tools["rmse_tb"]: rmse})
    writer.add_summary(summ, global_step=generation)
    writer.flush()


def create_tensorboard_tools():
    # create object for measurement of error
    rmse_tb = tf.placeholder(tf.float32, shape=[])
    unsupervised_mse_tb = tf.summary.scalar("streaking_trace_rmse", rmse_tb)

    tensorboard_tools = {}
    tensorboard_tools["rmse_tb"] = rmse_tb
    tensorboard_tools["unsupervised_mse_tb"] = unsupervised_mse_tb

    return tensorboard_tools



def get_trace(index, filename):


    with tables.open_file(filename, mode='r') as hdf5_file:

        # use the non noise trace
        trace = hdf5_file.root.trace[index,:]

        actual_params = {}

        actual_params['xuv_coefs'] = hdf5_file.root.xuv_coefs[index,:]

        actual_params['ir_params'] = hdf5_file.root.ir_params[index, :]

    return trace, actual_params




def calc_vecs_and_rmse(individual, measured_trace, tf_generator_graphs, sess, plot_and_graph=None):

    # append 0 for linear phase
    xuv_values = np.append([0], individual["xuv"][0])
    # print(xuv_values)

    ir_values = individual["ir"]
    # print(ir_values)

    generated_trace = sess.run(tf_generator_graphs["image"], feed_dict={tf_generator_graphs["xuv_coefs_in"]: xuv_values.reshape(1, -1),
                                                            tf_generator_graphs["ir_values_in"]: ir_values})

    # calculate rmse
    trace_rmse = np.sqrt(
        (1 / len(measured_trace)) * np.sum(
            (measured_trace - generated_trace.reshape(-1)) ** 2))

    if plot_and_graph is not None:

        # add tensorboard value
        add_tensorboard_values(trace_rmse, generation=plot_and_graph["generation"],
                               sess=sess, writer=plot_and_graph["writer"],
                               tensorboard_tools=plot_and_graph["tensorboard_tools"])




    return trace_rmse




def create_population(create_individual, n):

    # return a list as the population
    population = []
    for i in range(n):
        population.append(create_individual())

    return population




def evaluate(individual, measured_trace, tf_generator_graphs, sess):

    rmse = calc_vecs_and_rmse(individual, measured_trace, tf_generator_graphs, sess)

    return rmse




def create_individual():

    individual = creator.Individual()

    # random numbers betwwen -1 and 1
    xuv_values = (2 * np.random.rand(4) - 1.0).reshape(1, -1)
    individual["xuv"] = xuv_values

    # random numbers betwwen -1 and 1
    ir_values = (2 * np.random.rand(4) - 1.0).reshape(1, -1)
    individual["ir"] = ir_values

    return individual





def genetic_algorithm(generations, pop_size, run_name, tf_generator_graphs, measured_trace,
                      tensorboard_tools):

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
                        tools.mutGaussian(mutant[vector], mu=0.0, sigma=0.2, indpb=0.6)
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

            plot_and_graph = {}
            plot_and_graph["generation"] = g
            plot_and_graph["writer"] = writer
            plot_and_graph["tensorboard_tools"] = tensorboard_tools
            plot_and_graph["run_name"] = run_name

            calc_vecs_and_rmse(best_ind, measured_trace, tf_generator_graphs, sess, plot_and_graph=plot_and_graph)



        # return the rmse of final result
        best_ind = tools.selBest(pop, 1)[0]
        return calc_vecs_and_rmse(best_ind, input_data, frequency_space, spline_params, sess, axes=axes, generation=g,
                       writer=writer, tensorboard_tools=tensorboard_tools, run_name=run_name)



if __name__ == "__main__":


    tensorboard_tools = create_tensorboard_tools()

    measured_trace, actual_params = get_trace(index=1, filename="train3.hdf5")

    tf_generator_graphs, streak_params = network3.initialize_xuv_ir_trace_graphs()


    genetic_algorithm(generations=100, pop_size=4, run_name="test1", tf_generator_graphs=tf_generator_graphs,
                      measured_trace=measured_trace, tensorboard_tools=tensorboard_tools)



