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
from ir_spectrum import ir_spectrum
from xuv_spectrum import spectrum
import unsupervised_retrieval


def plot_image_and_fields(plot_and_graph,
                          predicted_fields,
                          predicted_streaking_trace, actual_streaking_trace,
                          rmse):

    actual_fields = plot_and_graph["actual_fields"]
    trace_c = len(plot_and_graph["streak_params"]["tau_values"])
    trace_r = len(plot_and_graph["streak_params"]["k_values"])


    plot_and_graph["plot_axes"]["actual_ir"].cla()
    plot_and_graph["plot_axes"]["actual_ir"].plot(ir_spectrum.fmat_cropped, np.abs(actual_fields["ir_f"])**2, color='black')
    plot_and_graph["plot_axes"]["actual_ir"].text(0.0, 1.1, "generation: {}".format(str(plot_and_graph["generation"])), transform=plot_and_graph["plot_axes"]["actual_ir"].transAxes)
    plot_and_graph["plot_axes"]["actual_ir"].text(0.0, 1.2, plot_and_graph["run_name"], transform=plot_and_graph["plot_axes"]["actual_ir"].transAxes)
    plot_and_graph["plot_axes"]["actual_ir_twinx"].cla()
    plot_and_graph["plot_axes"]["actual_ir_twinx"].text(0.0, 1.0, "actual_ir", backgroundcolor="white", transform=plot_and_graph["plot_axes"]["actual_ir_twinx"].transAxes)
    plot_and_graph["plot_axes"]["actual_ir_twinx"].plot(ir_spectrum.fmat_cropped, np.unwrap(np.angle(actual_fields["ir_f"])), color='green')
    plot_and_graph["plot_axes"]["actual_ir_twinx"].tick_params(axis='y', colors='green')

    plot_and_graph["plot_axes"]["actual_xuv"].cla()
    plot_and_graph["plot_axes"]["actual_xuv"].plot(spectrum.fmat_cropped, np.abs(actual_fields["xuv_f"]) ** 2, color='black')
    plot_and_graph["plot_axes"]["actual_xuv_twinx"].cla()
    plot_and_graph["plot_axes"]["actual_xuv_twinx"].text(0.0, 1.0, "actual_xuv", backgroundcolor="white", transform=plot_and_graph["plot_axes"]["actual_xuv_twinx"].transAxes)
    plot_and_graph["plot_axes"]["actual_xuv_twinx"].plot(spectrum.fmat_cropped, np.unwrap(np.angle(actual_fields["xuv_f"])), color='green')
    plot_and_graph["plot_axes"]["actual_xuv_twinx"].tick_params(axis='y', colors='green')

    # actual streaking trace
    plot_and_graph["plot_axes"]["actual_trace"].cla()
    plot_and_graph["plot_axes"]["actual_trace"].pcolormesh(plot_and_graph["streak_params"]["tau_values"], plot_and_graph["streak_params"]["k_values"], actual_streaking_trace.reshape(trace_r, trace_c), cmap='jet')
    plot_and_graph["plot_axes"]["actual_trace"].text(0.0, 1.0, "actual_trace", backgroundcolor="white", transform=plot_and_graph["plot_axes"]["actual_trace"].transAxes)
    plot_and_graph["plot_axes"]["actual_trace"].text(0.5, 1.0, "actual_trace", backgroundcolor="white",
                                                     transform=plot_and_graph["plot_axes"]["Genetic Algorithm"].transAxes)


    # plot predicted trace
    plot_and_graph["plot_axes"]["predicted_ir"].cla()
    plot_and_graph["plot_axes"]["predicted_ir"].plot(ir_spectrum.fmat_cropped, np.abs(predicted_fields["ir_f"].reshape(-1)) ** 2, color='black')
    plot_and_graph["plot_axes"]["predicted_ir_twinx"].cla()
    plot_and_graph["plot_axes"]["predicted_ir_twinx"].text(0.0, 1.0, "actual_ir", backgroundcolor="white",
                                 transform=plot_and_graph["plot_axes"]["actual_ir_twinx"].transAxes)
    plot_and_graph["plot_axes"]["predicted_ir_twinx"].plot(ir_spectrum.fmat_cropped, np.unwrap(np.angle(predicted_fields["ir_f"].reshape(-1))), color='green')
    plot_and_graph["plot_axes"]["predicted_ir_twinx"].tick_params(axis='y', colors='green')

    plot_and_graph["plot_axes"]["predicted_xuv"].cla()
    plot_and_graph["plot_axes"]["predicted_xuv"].plot(spectrum.fmat_cropped, np.abs(predicted_fields["xuv_f"].reshape(-1)) ** 2, color='black')
    plot_and_graph["plot_axes"]["predicted_xuv_twinx"].cla()
    plot_and_graph["plot_axes"]["predicted_xuv_twinx"].text(0.0, 1.1, "predicted_xuv", backgroundcolor="white",
                                  transform=plot_and_graph["plot_axes"]["predicted_xuv_twinx"].transAxes)
    plot_and_graph["plot_axes"]["predicted_xuv_twinx"].plot(spectrum.fmat_cropped, np.unwrap(np.angle(predicted_fields["xuv_f"].reshape(-1))), color='green')
    plot_and_graph["plot_axes"]["predicted_xuv_twinx"].tick_params(axis='y', colors='green')

    # predicted streaking trace
    plot_and_graph["plot_axes"]["predicted_trace"].cla()
    plot_and_graph["plot_axes"]["predicted_trace"].pcolormesh(plot_and_graph["streak_params"]["tau_values"], plot_and_graph["streak_params"]["k_values"], predicted_streaking_trace, cmap='jet')
    plot_and_graph["plot_axes"]["predicted_trace"].text(0.0, 1.0, "predicted_trace", backgroundcolor="white",
                              transform=plot_and_graph["plot_axes"]["predicted_trace"].transAxes)
    plot_and_graph["plot_axes"]["predicted_trace"].text(0.0, 0.1, "rmse: {}".format(str(round(rmse, 5))), backgroundcolor="white",
                                 transform=plot_and_graph["plot_axes"]["predicted_trace"].transAxes)


    if plot_and_graph["generation"] % 5 == 0 or plot_and_graph["generation"] == 1:
        #plt.savefig("./gapictures/{}.png".format(generation))

        dir = "./gapictures/" + plot_and_graph["run_name"] + "/"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        plt.savefig(dir + str(plot_and_graph["generation"]) + ".png")

    # save the raw files
    if plot_and_graph["generation"] % 100 == 0:

        with open("./gapictures/fields.p", "wb") as file:
            save_files = {}
            save_files["predicted_fields"] = predicted_fields
            save_files["actual_fields"] = actual_fields
            save_files["generation"] = plot_and_graph["generation"]
            pickle.dump(save_files,file)


    plt.pause(0.0001)



def plot_image_and_fields_exp(plot_and_graph,
                          predicted_fields,
                          predicted_streaking_trace, actual_streaking_trace,
                          rmse):


    trace_c = len(plot_and_graph["streak_params"]["tau_values"])
    trace_r = len(plot_and_graph["streak_params"]["k_values"])

    # actual streaking trace
    plot_and_graph["plot_axes"]["actual_trace"].cla()
    plot_and_graph["plot_axes"]["actual_trace"].pcolormesh(plot_and_graph["streak_params"]["tau_values"], plot_and_graph["streak_params"]["k_values"], actual_streaking_trace.reshape(trace_r, trace_c), cmap='jet')
    plot_and_graph["plot_axes"]["actual_trace"].text(0.0, 1.0, "actual_trace", backgroundcolor="white", transform=plot_and_graph["plot_axes"]["actual_trace"].transAxes)


    # plot predicted trace
    plot_and_graph["plot_axes"]["predicted_ir"].cla()
    plot_and_graph["plot_axes"]["predicted_ir"].plot(ir_spectrum.fmat_cropped, np.abs(predicted_fields["ir_f"].reshape(-1)) ** 2, color='black')
    plot_and_graph["plot_axes"]["predicted_ir_twinx"].cla()
    plot_and_graph["plot_axes"]["predicted_ir_twinx"].plot(ir_spectrum.fmat_cropped, np.unwrap(np.angle(predicted_fields["ir_f"].reshape(-1))), color='green')
    plot_and_graph["plot_axes"]["predicted_ir_twinx"].tick_params(axis='y', colors='green')

    # xuv f
    plot_and_graph["plot_axes"]["predicted_xuv"].cla()
    plot_and_graph["plot_axes"]["predicted_xuv"].plot(spectrum.fmat_cropped, np.abs(predicted_fields["xuv_f"].reshape(-1)) ** 2, color='black')
    plot_and_graph["plot_axes"]["predicted_xuv_twinx"].cla()
    plot_and_graph["plot_axes"]["predicted_xuv_twinx"].text(0.0, 1.0, "predicted_xuv", backgroundcolor="white",
                                  transform=plot_and_graph["plot_axes"]["predicted_xuv_twinx"].transAxes)
    plot_and_graph["plot_axes"]["predicted_xuv_twinx"].plot(spectrum.fmat_cropped, np.unwrap(np.angle(predicted_fields["xuv_f"].reshape(-1))), color='green')
    plot_and_graph["plot_axes"]["predicted_xuv_twinx"].tick_params(axis='y', colors='green')

    # xuv in time
    plot_and_graph["plot_axes"]["predicted_xuv_t"].cla()
    plot_and_graph["plot_axes"]["predicted_xuv_t"].plot(spectrum.tmat, np.abs(predicted_fields["xuv_t"].reshape(-1))**2, color="black")




    # predicted streaking trace
    plot_and_graph["plot_axes"]["generated_trace"].cla()
    plot_and_graph["plot_axes"]["generated_trace"].pcolormesh(plot_and_graph["streak_params"]["tau_values"], plot_and_graph["streak_params"]["k_values"], predicted_streaking_trace, cmap='jet')
    plot_and_graph["plot_axes"]["generated_trace"].text(0.0, 1.0, "generated_trace", backgroundcolor="white",
                              transform=plot_and_graph["plot_axes"]["generated_trace"].transAxes)
    plot_and_graph["plot_axes"]["generated_trace"].text(0.0, 0.1, "RMSE: {}".format(str(round(rmse, 3))), backgroundcolor="white",
                                 transform=plot_and_graph["plot_axes"]["generated_trace"].transAxes)


    if plot_and_graph["generation"] % 5 == 0 or plot_and_graph["generation"] == 1:
        #plt.savefig("./gapictures/{}.png".format(generation))

        dir = "./gapictures/" + plot_and_graph["run_name"] + "/"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        plt.savefig(dir + str(plot_and_graph["generation"]) + ".png")

    # save the raw files
    if plot_and_graph["generation"] % 100 == 0:

        with open("./gapictures/ga_fields.p", "wb") as file:
            save_files = {}
            save_files["predicted_fields"] = predicted_fields
            save_files["actual_streaking_trace"] = actual_streaking_trace
            save_files["predicted_streaking_trace"] = predicted_streaking_trace
            save_files["generation"] = plot_and_graph["generation"]
            pickle.dump(save_files,file)


    plt.pause(0.0001)

def create_plot_axes():
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(4, 2)
    # fig.subplots_adjust(hspace=0.9)

    axes = {}

    axes["actual_ir"] = fig.add_subplot(gs[0, 0])
    axes["actual_ir_twinx"] = axes["actual_ir"].twinx()
    axes["actual_xuv"] = fig.add_subplot(gs[0, 1])
    axes["actual_xuv_twinx"] = axes["actual_xuv"].twinx()

    axes["actual_trace"] = fig.add_subplot(gs[1, :])

    axes["predicted_ir"] = fig.add_subplot(gs[2, 0])
    axes["predicted_ir_twinx"] = axes["predicted_ir"].twinx()
    axes["predicted_xuv"] = fig.add_subplot(gs[2, 1])
    axes["predicted_xuv_twinx"] = axes["predicted_xuv"].twinx()

    axes["predicted_trace"] = fig.add_subplot(gs[3, :])

    return axes

def create_exp_plot_axes():

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1)
    gs = fig.add_gridspec(3, 3)

    axes_dict = {}
    axes_dict["actual_trace"] = fig.add_subplot(gs[0, :])

    axes_dict["predicted_xuv_t"] = fig.add_subplot(gs[1, 2])

    axes_dict["predicted_xuv"] = fig.add_subplot(gs[1, 1])
    axes_dict["predicted_xuv_twinx"] = axes_dict["predicted_xuv"].twinx()

    axes_dict["predicted_ir"] = fig.add_subplot(gs[1, 0])
    axes_dict["predicted_ir_twinx"] = axes_dict["predicted_ir"].twinx()

    axes_dict["generated_trace"] = fig.add_subplot(gs[2, :])

    return axes_dict

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
        #trace = hdf5_file.root.trace[index,:]
        trace = hdf5_file.root.noise_trace[index,:]

        actual_params = {}

        actual_params['xuv_coefs'] = hdf5_file.root.xuv_coefs[index,:]

        actual_params['ir_params'] = hdf5_file.root.ir_params[index, :]

    return trace, actual_params

def calc_vecs_and_rmse(individual, measured_trace, tf_generator_graphs, sess, plot_and_graph=None):

    # append 0 for linear phase
    xuv_values = np.append([0], individual["xuv"])
    # print(xuv_values)

    ir_values = individual["ir"]
    # print(ir_values)

    generated_trace = sess.run(tf_generator_graphs["image"], feed_dict={tf_generator_graphs["xuv_coefs_in"]: xuv_values.reshape(1, -1),
                                                            tf_generator_graphs["ir_values_in"]: ir_values.reshape(1, -1)})

    # calculate rmse
    trace_rmse = np.sqrt(
        (1 / len(measured_trace.reshape(-1))) * np.sum(
            (measured_trace - generated_trace.reshape(-1)) ** 2))

    if plot_and_graph is not None:

        # add tensorboard value
        add_tensorboard_values(trace_rmse, generation=plot_and_graph["generation"],
                               sess=sess, writer=plot_and_graph["writer"],
                               tensorboard_tools=plot_and_graph["tensorboard_tools"])


        if "actual_fields" in plot_and_graph:
            # simulated trace
            predicted_fields = {}
            predicted_fields["ir_f"] = sess.run(tf_generator_graphs["ir_E_prop"]["f_cropped"],
                                                feed_dict={tf_generator_graphs["ir_values_in"]: ir_values.reshape(1, -1)})
            predicted_fields["xuv_f"] = sess.run(tf_generator_graphs["xuv_E_prop"]["f_cropped"],
                                                 feed_dict={tf_generator_graphs["xuv_coefs_in"]: xuv_values.reshape(1, -1)})
            # plot
            plot_image_and_fields(plot_and_graph=plot_and_graph,
                                  predicted_fields=predicted_fields,
                                  predicted_streaking_trace=generated_trace,
                                  actual_streaking_trace=measured_trace,
                                  rmse=trace_rmse)
        else:
            # experimental trace
            predicted_fields = {}
            predicted_fields["ir_f"] = sess.run(tf_generator_graphs["ir_E_prop"]["f_cropped"],
                                                feed_dict={
                                                    tf_generator_graphs["ir_values_in"]: ir_values.reshape(1, -1)})
            predicted_fields["xuv_f"] = sess.run(tf_generator_graphs["xuv_E_prop"]["f_cropped"],
                                                 feed_dict={
                                                     tf_generator_graphs["xuv_coefs_in"]: xuv_values.reshape(1, -1)})
            predicted_fields["xuv_t"] = sess.run(tf_generator_graphs["xuv_E_prop"]["t"],
                                                 feed_dict={
                                                     tf_generator_graphs["xuv_coefs_in"]: xuv_values.reshape(1, -1)})

            plot_image_and_fields_exp(plot_and_graph=plot_and_graph,
                                  predicted_fields=predicted_fields,
                                  predicted_streaking_trace=generated_trace,
                                  actual_streaking_trace=measured_trace,
                                  rmse=trace_rmse)






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
    xuv_values = (2 * np.random.rand(4) - 1.0)
    individual["xuv"] = xuv_values

    # random numbers betwwen -1 and 1
    ir_values = (2 * np.random.rand(4) - 1.0)
    individual["ir"] = ir_values

    return individual

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

    tensorboard_tools = create_tensorboard_tools()
    tf_generator_graphs, streak_params = network3.initialize_xuv_ir_trace_graphs()





    #..................................
    #.... retrieve simulated trace.....
    #..................................

    # plot_axes = create_plot_axes()
    # plot_and_graph = {}
    # plot_and_graph["plot_axes"] = plot_axes
    # plot_and_graph["streak_params"] = streak_params
    # measured_trace, actual_params = get_trace(index=3, filename="train3.hdf5")
    # # get the actual fields (from params)
    # with tf.Session() as sess:
    #     plot_and_graph["actual_fields"] = {}
    #     plot_and_graph["actual_fields"]["ir_f"] = sess.run(tf_generator_graphs["ir_E_prop"]["f_cropped"],
    #                                                        feed_dict={
    #                                                            tf_generator_graphs["ir_values_in"]: actual_params[
    #                                                                "ir_params"].reshape(1, -1)}
    #                                                        ).reshape(-1)
    #     plot_and_graph["actual_fields"]["xuv_f"] = sess.run(tf_generator_graphs["xuv_E_prop"]["f_cropped"],
    #                                                         feed_dict={
    #                                                             tf_generator_graphs["xuv_coefs_in"]: actual_params[
    #                                                                 "xuv_coefs"].reshape(1, -1)}
    #                                                         ).reshape(-1)




    #..................................
    # .....retrieve measured trace.....
    #..................................
    plot_axes = create_exp_plot_axes()
    plot_and_graph = {}
    plot_and_graph["plot_axes"] = plot_axes
    plot_and_graph["streak_params"] = streak_params
    _, _, measured_trace = unsupervised_retrieval.get_measured_trace()
    measured_trace = measured_trace.reshape(1, -1)



    genetic_algorithm(generations=500, pop_size=500, run_name="experimental_retrieval1", tf_generator_graphs=tf_generator_graphs,
                      measured_trace=measured_trace, tensorboard_tools=tensorboard_tools, plot_and_graph=plot_and_graph)
