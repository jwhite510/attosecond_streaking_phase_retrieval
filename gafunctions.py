import numpy as np
import matplotlib.pyplot as plt
import random
from deap import base
from deap import creator
from deap import tools
import os
#os.chdir('../')
import tensorflow as tf
import unsupervised
import crab_tf2
from scipy.interpolate import BSpline
import pickle




def add_tensorboard_values(rmse, generation):

    summ = sess.run(unsupervised_mse_tb, feed_dict={rmse_tb: rmse})
    writer.add_summary(summ, global_step=generation)
    writer.flush()
    pass




def calc_vecs_and_rmse(individual, plotting, generation=None):
    # a = np.sum(individual["ir_amplitude"])
    # b = np.sum(individual["ir_phase"])
    # c = np.sum(individual["xuv_phase"])

    # interpolate the coef values onto the fmat
    xuv_phase_spl = BSpline(xuv_knot_locations, individual["xuv_phase"], k)
    ir_amp_spl = BSpline(ir_knot_locations, individual["ir_amplitude"], k)
    ir_phase_spl = BSpline(ir_knot_locations, individual["ir_phase"], k)

    xuv_phase = xuv_phase_spl(xuv_fmat)
    ir_amp = ir_amp_spl(ir_fmat)
    ir_phase = ir_phase_spl(ir_fmat)

    # construct complex field vectors from phase and amplitude curve
    xuv_vec = xuv_amp * np.exp(1j * xuv_phase)
    ir_vec = ir_amp * np.exp(1j * ir_phase)

    # compare to measured streaking trace
    image_out = sess.run(crab_tf2.image, feed_dict={crab_tf2.ir_cropped_f: ir_vec, crab_tf2.xuv_cropped_f: xuv_vec})

    # calculate rmse
    trace_rmse = np.sqrt(
        (1 / len(image_out.reshape(-1))) * np.sum((actual_trace.reshape(-1) - image_out.reshape(-1)) ** 2))


    if plotting:
        predicted_fields = {}
        predicted_fields["ir_phase"] = ir_phase
        predicted_fields["ir_amp"] = ir_amp
        predicted_fields["xuv_phase"] = xuv_phase
        predicted_fields["xuv_amp"] = xuv_amp
        plot_image_and_fields(axes=plot_axes, predicted_fields=predicted_fields, actual_fields=actual_fields,
                              xuv_fmat=xuv_fmat, ir_fmat=ir_fmat, predicted_streaking_trace=image_out,
                              actual_streaking_trace=actual_trace, generation=generation, rmse=trace_rmse)

        add_tensorboard_values(trace_rmse, generation)



    return trace_rmse






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



def plot_image_and_fields(axes, predicted_fields, actual_fields, xuv_fmat, ir_fmat,
                          predicted_streaking_trace, actual_streaking_trace, generation, rmse):
    axes["actual_ir"].cla()
    axes["actual_ir"].plot(ir_fmat, np.abs(actual_fields["ir_f"])**2, color='black')
    axes["actual_ir"].text(0.0, 1.1, "generation: {}".format(str(generation)), transform=axes["actual_ir"].transAxes)
    axes["actual_ir_twinx"].cla()
    axes["actual_ir_twinx"].text(0.0, 0.9, "actual_ir", backgroundcolor="white", transform=axes["actual_ir_twinx"].transAxes)
    axes["actual_ir_twinx"].plot(ir_fmat, np.unwrap(np.angle(actual_fields["ir_f"])), color='green')
    axes["actual_ir_twinx"].tick_params(axis='y', colors='green')

    axes["actual_xuv"].cla()
    axes["actual_xuv"].plot(xuv_fmat, np.abs(actual_fields["xuv_f"]) ** 2, color='black')
    axes["actual_xuv_twinx"].cla()
    axes["actual_xuv_twinx"].text(0.0, 0.9, "actual_xuv", backgroundcolor="white", transform=axes["actual_xuv_twinx"].transAxes)
    axes["actual_xuv_twinx"].plot(xuv_fmat, np.unwrap(np.angle(actual_fields["xuv_f"])), color='green')
    axes["actual_xuv_twinx"].tick_params(axis='y', colors='green')

    # actual streaking trace
    axes["actual_trace"].cla()
    axes["actual_trace"].pcolormesh(crab_tf2.tau_values, crab_tf2.k_values, actual_streaking_trace, cmap='jet')
    axes["actual_trace"].text(0.0, 0.9, "actual_trace", backgroundcolor="white", transform=axes["actual_trace"].transAxes)


    # predicted phase crop
    xuv_left_crop, xuv_right_crop = 10, 10
    ir_left_crop, ir_right_crop = 3, 3
    axes["predicted_ir"].cla()
    axes["predicted_ir"].plot(ir_fmat, predicted_fields["ir_amp"], color='black')
    axes["predicted_ir_twinx"].cla()
    axes["predicted_ir_twinx"].text(0.0, 0.9, "predicted_ir", backgroundcolor="white", transform=axes["predicted_ir_twinx"].transAxes)
    axes["predicted_ir_twinx"].plot(ir_fmat[ir_left_crop:-ir_right_crop], predicted_fields["ir_phase"][ir_left_crop:-ir_right_crop], color='green')
    axes["predicted_ir_twinx"].tick_params(axis='y', colors='green')

    axes["predicted_xuv"].cla()
    axes["predicted_xuv"].plot(xuv_fmat, predicted_fields["xuv_amp"]**2, color='black')
    axes["predicted_xuv_twinx"].cla()
    axes["predicted_xuv_twinx"].text(0.0, 0.9, "predicted_xuv", backgroundcolor="white",transform=axes["predicted_xuv_twinx"].transAxes)
    axes["predicted_xuv_twinx"].plot(xuv_fmat[xuv_left_crop:-xuv_right_crop], predicted_fields["xuv_phase"][xuv_left_crop:-xuv_right_crop], color='green')
    axes["predicted_xuv_twinx"].tick_params(axis='y', colors='green')

    # predicted streaking trace
    axes["predicted_trace"].cla()
    axes["predicted_trace"].pcolormesh(crab_tf2.tau_values, crab_tf2.k_values, predicted_streaking_trace, cmap='jet')
    axes["predicted_trace"].text(0.0, 0.9, "predicted_trace", backgroundcolor="white",
                              transform=axes["predicted_trace"].transAxes)
    axes["predicted_trace"].text(0.0, 0.1, "rmse: {}".format(str(round(rmse, 5))), backgroundcolor="white",
                                 transform=axes["predicted_trace"].transAxes)

    if generation % 10 == 0 or generation == 1:
        plt.savefig("./gapictures/{}.png".format(generation))

        dir = "./gapictures/" + run_name + "/"
        if not os.path.isdir(dir):
            os.makedirs(dir)
        plt.savefig(dir + str(generation) + ".png")

    plt.pause(0.0001)






def create_individual():

    individual = creator.Individual()

    #individual["ir_amplitude"] = np.random.rand(3)
    #individual["ir_phase"] = np.random.rand(3)
    #individual["xuv_phase"] = np.random.rand(3)
    #return individual


    individual["ir_amplitude"] = 1.0*np.random.rand(ir_points_length)
    individual["ir_amplitude"][0:8] = 0.0
    individual["ir_amplitude"][-8:] = 0.0
    individual["ir_phase"] = 1.0*np.random.rand(ir_points_length)
    individual["xuv_phase"] = 10.0*np.random.rand(xuv_points_length)

    # calc_vecs_and_rmse(individual, plotting=True)
    # plt.ioff()
    # plt.show()

    return individual


def create_population(create_individual, n):

    # return a list as the population
    population = []
    for i in range(n):
        population.append(create_individual())

    return population


def evaluate(individual):


    rmse = calc_vecs_and_rmse(individual, plotting=False)

    return rmse


def generate_ir_xuv_complex_fields(ir_phi, ir_amp, xuv_phi, knot_values):

    # define the curves with these coefficients

    # knot values for ir, xuv are defined prior

    # return a complex field vector matching the input of the tensorflownet
    return None



def genetic_algorithm(generations, pop_size):
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", dict, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("create_individual", create_individual)
    toolbox.register("create_population", create_population, toolbox.create_individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament, tournsize=4)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)

    # create the initial population
    pop = toolbox.create_population(n=pop_size)
    # print(pop[0].fitness.values)

    # evaluate and assign fitness numbers
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit,

    print("  Evaluated %i individuals" % len(pop))

    # fits = [ind.fitness.values[0] for ind in pop]

    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB, MUTPB2 = 0.2, 0.2, 0.5
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
            for vector in ['ir_phase', 'xuv_phase', 'ir_amplitude']:
                if random.random() < CXPB:
                    toolbox.mate(child1[vector], child2[vector])
                    re_evaluate = True
            if re_evaluate:
                del child1.fitness.values
                del child2.fitness.values


        for mutant in offspring:
            # mutate an individual with probability MUTPB
            re_evaluate = False
            for vector in ['ir_phase', 'xuv_phase', 'ir_amplitude']:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant[vector])
                    re_evaluate = True
            if re_evaluate:
                del mutant.fitness.values


        for mutant in offspring:
            # mutate an individual with probabililty MUTPB2
            re_evaluate = False
            for vector in ['ir_phase', 'xuv_phase', 'ir_amplitude']:
                if random.random() < MUTPB2:
                    # tools.mutGaussian(mutant[vector], mu=0.0, sigma=0.2, indpb=0.2)
                    tools.mutGaussian(mutant[vector], mu=0.0, sigma=5.0, indpb=0.2)
                    re_evaluate = True
            if re_evaluate:
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
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
        calc_vecs_and_rmse(best_ind, plotting=True, generation=g)







if __name__ == "__main__":


    run_name = 'run6'

    ir_points_length = 20
    xuv_points_length = 50

    # define global variables for vector calculation
    # knot locations
    xuv_fmat = crab_tf2.xuv.f_cropped
    ir_fmat = crab_tf2.ir.f_cropped
    # order
    k = 5

    # define number of knots
    xuv_knots = xuv_points_length + k + 1
    ir_knots = ir_points_length + k + 1


    # knot locations
    xuv_knot_locations = np.linspace(xuv_fmat[0], xuv_fmat[-1], xuv_knots)
    ir_knot_locations = np.linspace(ir_fmat[0], ir_fmat[-1], ir_knots)

    # get xuv field amplitude
    with open('measured_spectrum.p', 'rb') as file:
        spectrum_data = pickle.load(file)
    xuv_sample = crab_tf2.XUV_Field(random_phase_taylor={'coefs': 15, 'amplitude': 0},
                                    measured_spectrum=spectrum_data)
    xuv_amp = np.abs(xuv_sample.Ef_prop_cropped)


    # create object for measurement of error
    rmse_tb = tf.placeholder(tf.float32, shape=[])
    unsupervised_mse_tb = tf.summary.scalar("streaking_trace_rmse", rmse_tb)




    # retrieve the trace and actual fields
    # actual_trace, actual_fields = unsupervised.get_trace(index=2, filename='attstrace_train2_processed.hdf5', plotting=False)
    actual_trace, actual_fields = unsupervised.get_trace(index=2, filename='attstrace_train2.hdf5', plotting=False)
    actual_trace = actual_trace.reshape(len(crab_tf2.p_values), len(crab_tf2.tau_values))

    # create axes for plotting
    plt.ion()
    plot_axes = create_plot_axes()


    with tf.Session() as sess:

        writer = tf.summary.FileWriter("./tensorboard_graph_ga/" + run_name)

        # run the genetic algorithm
        genetic_algorithm(generations=5000, pop_size=5000)



