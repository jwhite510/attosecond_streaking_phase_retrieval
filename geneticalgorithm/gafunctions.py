import numpy as np
import random
from deap import base
from deap import creator
from deap import tools




def create_individual():

    individual = creator.Individual()
    individual["ir_amplitude"] = np.array([1, 2 ,3])
    individual["ir_phase"] = np.array([1, 2 ,3])
    individual["xuv_phase"] = np.array([1, 2 ,3])

    return individual


def create_population(create_individual, n):

    # return a list as the population
    population = []
    for i in range(n):
        population.append(create_individual())

    return population


def evaluate(individual):
    a = np.sum(individual["ir_amplitude"])
    b = np.sum(individual["ir_phase"])
    c = np.sum(individual["xuv_phase"])
    return a + b + c


def generate_ir_xuv_complex_fields(ir_phi, ir_amp, xuv_phi, knot_values):

    # define the curves with these coefficients

    # knot values for ir, xuv are defined prior

    # return a complex field vector matching the input of the tensorflownet
    return None





creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", dict, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("create_individual", create_individual)
toolbox.register("create_population", create_population, toolbox.create_individual)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("mate", tools.cxTwoPoint)



# create the initial population
pop = toolbox.create_population(n=5)
print(pop[0].fitness.values)


# evaluate and assign fitness numbers
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit,

print("  Evaluated %i individuals" % len(pop))

fits = [ind.fitness.values[0] for ind in pop]


# MUTPB is the probability for mutating an individual
# CXPB, MUTPB, MUTPB2 = 0.2, 0.2, 0.5
CXPB, MUTPB, MUTPB2 = 1.0, 0.2, 0.5



# Variable keeping track of the number of generations
g = 0

generations = 10
while g <= generations:
    g = g + 1
    print("-- Generation %i --" % g)

    offspring = toolbox.select(pop, len(pop))

    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):

        print("bitch")
        print(child1)

        # cross two individuals with probability CXPB
        if random.random() < CXPB:
            toolbox.mate(child1, child2)

            # fitness values of the children
            # must be recalculated later
            del child1.fitness.values
            del child2.fitness.values

            print("bitch")
            print(child1)
            exit(0)

    exit(0)


    for mutant in offspring:

        # mutate an individual with probability MUTPB
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    for mutant in offspring:

        # mutate an individual with probabililty MUTPB2
        if random.random() < MUTPB2:
            # print('before: ', mutant)
            tools.mutGaussian(mutant, mu=0.0, sigma=0.2, indpb=0.2)
            # print('after: ', mutant)
            # exit(0)
            del mutant.fitness.values






