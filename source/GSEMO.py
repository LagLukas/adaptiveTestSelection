try:
    from source.set_cover import *
    from source.gcais_population import GCAISPopulation
except Exception as _:
    from set_cover import *
    from gcais_population import GCAISPopulation
import numpy as np
import copy
import random
import sys
import time
import os


class Simple_GSEMO:

    def __init__(self, problem_instance, loc_pops, iterations):
        if loc_pops == 1:
            self.name = "SIMPLE_SEMO"
        else:
            self.name = "SIMPLE_GSEMO"
        self.populations = []
        self.problem = problem_instance
        self.iterations = iterations
        self.pops = loc_pops
        prob_shape = self.problem.problem_instance.shape
        self.send_prob = loc_pops / (prob_shape[0] * prob_shape[1])
        number_of_sets = self.problem.problem_instance.shape[0]
        for _ in range(0, loc_pops):
            sol_vector = np.ones(number_of_sets)
            population = [Solution(self.problem, sol_vector)]
            self.populations.append(population)
        self.iter = 0

    def save_pop_results(self):
        pop = []
        for population in self.populations:
            pop.extend(population)
        GCAISPopulation.convert_and_save("PARETO_FRONTIERS" + os.sep + self.logger.get_file_name(), self.problem, pop)

    def superior(self, sol_a, sol_b):
        if sol_a.cost <= sol_b.cost:
            if sum(sol_a.covered_elements) > sum(sol_b.covered_elements):
                return True
        if sum(sol_a.covered_elements) >= sum(sol_b.covered_elements):
            if sol_a.cost < sol_b.cost:
                return True
        return False

    def mutate(self, sol):
        mut_prob = 1.0 / len(sol.set_vector)
        new_sol = copy.deepcopy(sol.set_vector)
        for i in range(0, len(sol.set_vector)):
            if random.random() < mut_prob:
                new_sol[i] = 1 if new_sol[i] == 0 else 0
        return Solution(self.problem, new_sol)

    def rand_choice(self, pop):
        i = random.randint(0, len(pop) - 1)
        return pop[i]

    def insert_to_pop(self, ele, pop):
        for other in pop:
            if self.superior(other, ele):
                return False
        pop.append(ele)
        return True

    def iteration(self):
        pareto_changed = False
        for pop in self.populations:
            ele = self.rand_choice(pop)
            mutated = self.mutate(ele)
            inserted = self.insert_to_pop(mutated, pop)
            if inserted:
                if random.random() < self.send_prob:
                    for other_pop in self.populations:
                        pareto_changed = pareto_changed or self.insert_to_pop(mutated, other_pop)
        self.iter = self.iter + 1
        return pareto_changed

    def set_logging(self, logger):
        self.logger = logger

    def get_population_size(self):
        size = 0
        for pop in self.populations:
            size += len(pop)
        return size

    def find_approximation(self):
        print("start GSEMO")
        found = 0
        old_best = sys.maxsize
        s = time.time()
        for i in range(0, self.iterations):
            if i % 10 == 0:
                print_now()
                print("finished " + str(i / self.iterations) + " percent of GSEMO")
            iter_start = time.time()
            pareto_changed = False
            try:
                pareto_changed = self.iteration()
            except Exception as _:
                break
            iter_end = time.time()
            feasible = []
            for pop in self.populations:
                feasible.extend(list(filter(lambda x: x.is_feasible, pop)))
            try:
                best = min(feasible, key=lambda x: x.cost).cost
            except Exception as e:
                best = sys.maxsize
            if pareto_changed:
                found = i
            if has_converged(found, i, time.time() - s):
                break
            self.logger.log_entry(self.iter, best, float(iter_end - iter_start), self.get_population_size())
        feasible = []
        for pop in self.populations:
            feasible.extend(list(filter(lambda x: x.is_feasible, pop)))
        if len(feasible) == 0:
            all_sets = np.ones(self.problem.problem_instance.shape[0])
            return Solution(self.problem, all_sets)
        print("finished GSEMO")
        self.save_pop_results()
        return min(feasible, key=lambda x: x.cost)


class GSEMO:

    BORDER = 200
    # to tackle exploding population
    CROPPING = False

    def __init__(self, problem_instance, loc_pops, iterations):
        self.name = "GSEMO"
        self.populations = []
        self.problem = problem_instance
        self.iterations = iterations
        self.pops = loc_pops
        prob_shape = self.problem.problem_instance.shape
        self.send_prob = loc_pops / (prob_shape[0] * prob_shape[1])
        number_of_sets = self.problem.problem_instance.shape[0]
        sol_vector = np.zeros(number_of_sets)
        for _ in range(0, loc_pops):
            population = {}
            sol_vector = np.zeros(number_of_sets)
            population[0] = {}
            population[0]["sols"] = [Solution(self.problem, sol_vector)]
            population[0]["best_covered"] = 0
            self.populations.append(population)
        self.iter = 0

    def superior(self, sol_a, sol_b):
        if sol_a.cost <= sol_b.cost:
            if sum(sol_a.covered_elements) > sum(sol_b.covered_elements):
                return True
        if sum(sol_a.covered_elements) >= sum(sol_b.covered_elements):
            if sol_a.cost < sol_b.cost:
                return True
        return False

    def mutate(self, sol):
        mut_prob = 1.0 / len(sol.set_vector)
        new_sol = copy.deepcopy(sol.set_vector)
        for i in range(0, len(sol.set_vector)):
            if random.random() < mut_prob:
                new_sol[i] = 1 if new_sol[i] == 0 else 0
        return Solution(self.problem, new_sol)

    def rand_choice(self, pop):
        keys = list(map(lambda x: int(x), pop.keys()))
        sizes = list(map(lambda x: len(pop[x]["sols"]), keys))
        total_size = sum(sizes)
        i = random.randint(0, total_size - 1)
        counter = 0
        for subpop_size in sizes:
            if i < subpop_size:
                break
            counter += 1
            i = i - subpop_size
        return pop[keys[counter]]["sols"][i]

    def insert_to_pop(self, ele, pop):
        try:
            # case worse solution
            if pop[ele.cost]["best_covered"] > ele.covered:
                return False
            # case equal solution
            if pop[ele.cost]["best_covered"] == ele.covered:
                for other in pop[ele]["sols"]:
                    if ele.equals_other_sol(other):
                        return False
                if len(pop[ele]["sols"]) < GSEMO.BORDER or not GSEMO.CROPPING:
                    pop[ele]["sols"].append(ele)
                    return True
                return False
            # case superior solution
            pop[ele.cost]["best_covered"] = ele.covered
            pop[ele.cost]["sols"] = [ele]
            max_key = max(list(map(lambda x: int(x), self.population.keys())))
            for i in range(ele.cost + 1, max_key):
                try:
                    if pop[i]["best_covered"] < ele.covered:
                        del pop[i]
                except Exception as _:
                    pass
            return True
        except Exception as _:
            pop[ele.cost] = {}
            pop[ele.cost]["best_covered"] = ele.covered
            pop[ele.cost]["sols"] = [ele]
            return True

    def iteration(self):
        for pop in self.populations:
            ele = self.rand_choice(pop)
            mutated = self.mutate(ele)
            inserted = self.insert_to_pop(mutated, pop)
            if inserted:
                if random.random() < self.send_prob:
                    for other_pop in self.populations:
                        self.insert_to_pop(mutated, other_pop)
        self.iter = self.iter + 1

    def set_logging(self, logger):
        self.logger = logger

    def find_approximation(self):
        print("start GSEMO")
        found = 0
        old_best = sys.maxsize
        s = time.time()
        for i in range(0, self.iterations):
            if i % 10 == 0:
                print_now()
                print("finished " + str(i / self.iterations) + " percent of GSEMO")
            iter_start = time.time()
            self.iteration()
            iter_end = time.time()
            feasible = []
            for pop in self.populations:
                for key in pop.keys():
                    feasible.extend(list(filter(lambda x: x.is_feasible, pop[key]["sols"])))
            try:
                best = min(feasible, key=lambda x: x.cost).cost
            except Exception as e:
                best = sys.maxsize
            if best != sys.maxsize:
                if best < old_best:
                    found = i
                    old_best = best
                else:
                    if has_converged(found, i, time.time() - s):
                        break
            self.logger.log_entry(self.iter, best, float(iter_end - iter_start))
        feasible = []
        for pop in self.populations:
            for key in pop.keys():
                feasible.extend(list(filter(lambda x: x.is_feasible, pop[key]["sols"])))
        if len(feasible) == 0:
            all_sets = np.ones(self.problem.problem_instance.shape[0])
            return Solution(self.problem, all_sets)
        print("finished GSEMO")
        return min(feasible, key=lambda x: x.cost)

if __name__ == "__main__":
    from source.set_cover import SteinerTriple
    from source.mylogging import Logging
    prob = SteinerTriple()
    eGCAIS = Simple_GSEMO(prob, 5, 30000)
    logger = Logging("fu.json")
    eGCAIS.set_logging(logger)
    print(eGCAIS.find_approximation())
    logger.save()
