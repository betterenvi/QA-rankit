import copy, random
from deap import base
from deap import creator
from deap import tools

class MyGA(object):
    def __init__(self, gen_param, evaluate, mut_indiv, CXPB=0.5, MUTPB=0.2, objective='FitnessMax'):
        creator.create(objective, base.Fitness, weights=(1.0 if objective=='FitnessMax' else -1.0,))
        creator.create("Individual", list, fitness=eval('creator.' + objective))
        self.toolbox = base.Toolbox()
        self.toolbox.register('gen_param', gen_param)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.gen_param)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", mut_indiv, indiv_pb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.all_gens = list()
        self.CXPB = CXPB
        self.MUTPB = MUTPB
        self.NGEN = 0
        self.best, self.best_fitness = None, None
        self.cmp_fitness = lambda x, y : x < y if objective=='FitnessMax' else x > y

    def init_pop(self, NPOP=10):
        print 'init population', NPOP
        self.NPOP = NPOP
        pop = self.toolbox.population(n=self.NPOP)
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        self.pop = pop
        self.all_gens.append(copy.deepcopy(pop))
        self.best, self.best_fitness = self.get_best()
        print "* Best", self.best_fitness, self.best

    def get_best(self, igen=-1):
        pop = self.all_gens[igen]
        best, best_fitness = pop[1], pop[1].fitness.values
        for indiv in pop[1:]:
            if self.cmp_fitness(best_fitness, indiv.fitness.values):
                best_fitness = indiv.fitness.values
                best = indiv
        return best, best_fitness
    
    def update_best(self):
        ibest, ibest_fitness = self.get_best(igen=-1)
        if self.cmp_fitness(self.best_fitness, ibest_fitness):
            self.best_fitness = ibest_fitness
            self.best = ibest
        return self

    def iterate(self, NGEN=10):
        self.NGEN += NGEN
        for g in range(NGEN):
            print("-- Generation %i --" % g)
            pop = self.pop
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
            self.all_gens.append(copy.deepcopy(pop))
            self.update_best()
            self.pop = pop
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
            
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
            print "* Best", self.best_fitness, self.best
