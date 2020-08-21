try:
    from source.population_initializer import PopulationCreator
    from source.greedy import GreedyAlgorithm
    from source.set_cover import *
    from source.gcais_population import GCAISPopulation
except Exception as _:
    from population_initializer import PopulationCreator
    from greedy import GreedyAlgorithm
    from set_cover import *
    from gcais_population import GCAISPopulation
import numpy as np
import copy
import random
import sys
import time
import os
import unittest


class BoundedGCAIS:

    def __init__(self, problem_instance, iterations, border, adaptive, joshi, with_rand=False):
        val_border = str(border) if border != sys.maxsize else "no_border"
        self.del_entry = False
        self.with_rand = with_rand
        if adaptive:
            self.name = "bounded_GCAIS_adaptive" + "_" + val_border
            if joshi:
                self.name = "bounded_GCAIS_adaptive" + "_" + val_border + "_joshi"
                if with_rand:
                    self.name += "_with_rand"
        else:
            self.name = "bounded_GCAIS" + "_" + val_border
            if with_rand:
                self.name += "_with_rand"
        self.problem_instance = problem_instance
        self.iterations = iterations
        self.border = border
        to_be_covered = self.problem_instance.problem_instance.shape[1]
        number_of_sets = self.problem_instance.problem_instance.shape[0]
        sol_vector = np.ones(number_of_sets)
        initial_sol = Solution(self.problem_instance, sol_vector)
        self.population = GCAISPopulation(initial_sol, to_be_covered, border, not self.del_entry, self.with_rand, problem_instance)
        self.iter = 0
        self.adapted = False
        self.adaptive = adaptive
        self.joshi = joshi

    def adapt_to_new_problem(self, new_prob_instance, deleted_sets):
        self.adapted = True
        self.problem_instance = new_prob_instance
        self.iter = 0
        self.population.adapt_to_new_problem(new_prob_instance, deleted_sets, self.problem_instance.problem_instance.shape[1], self.joshi)

    def mutate(self, sol):
        mut_prob = 1.0 / len(sol.set_vector)
        new_sol = copy.deepcopy(sol.set_vector)
        for i in range(0, len(sol.set_vector)):
            if random.random() < mut_prob:
                new_sol[i] = 1 if new_sol[i] == 0 else 0
        return Solution(self.problem_instance, new_sol)

    def mutate_and_insert(self):
        pareto_changed = False
        mutated = []
        for key in self.population.table.keys():
            chunk = self.population.table[key]["sols"]
            mutated.append(list(map(lambda x: self.mutate(x), chunk)))
        for mutations in mutated:
            for mutated in mutations:
                pareto_changed = pareto_changed or self.population.try_insert(mutated)
        return pareto_changed

    def iteration(self):
        pareto_changed = self.mutate_and_insert()
        self.population.clean()
        self.iter = self.iter + 1
        return pareto_changed

    def set_logging(self, logger):
        self.logger = logger

    def find_approximation(self):
        print("start " + self.name)
        old_best = sys.maxsize
        found = -1
        s = time.time()
        for i in range(0, self.iterations):
            if i % 10 == 0:
                print_now()
                print("finished " + str(i / self.iterations) + " percent of " + self.name)
            iter_start = time.time()
            pareto_changed = self.iteration()
            iter_end = time.time()
            if pareto_changed:
                found = i
            try:
                best = self.population.get_best_cost()
            except Exception as _:
                best = sys.maxsize
            if has_converged(found, self.iter, time.time() - s, self.adapted):
                break
            self.logger.log_entry(self.iter, best, float(iter_end - iter_start), self.population.get_population_size())
        print("finished " + self.name)
        self.population.save_rough_info("PARETO_FRONTIERS" + os.sep + self.logger.get_file_name(), self.problem_instance)
        return self.population.get_best()


class GCAIS_BASE:

    def __init__(self, problem_instance, iterations):
        self.name = "GCAIS_BASE"
        self.population = {}
        self.problem_instance = problem_instance
        self.iterations = iterations
        number_of_sets = self.problem_instance.problem_instance.shape[0]
        sol_vector = np.ones(number_of_sets)
        self.population = [Solution(self.problem_instance, sol_vector)]
        self.iter = 0

    def superior(self, sol_a, sol_b):
        if sol_a.cost <= sol_b.cost:
            if sol_a.covered > sol_b.covered:
                return True
        if sol_a.covered >= sol_b.covered:
            if sol_a.cost < sol_b.cost:
                return True
        return False

    def dominated_by_any(self, sol_set, ele):
        for solution in sol_set:
            if self.superior(solution, ele):
                return True
        return False

    def mutate(self, sol):
        mut_prob = 1.0 / len(sol.set_vector)
        new_sol = copy.deepcopy(sol.set_vector)
        for i in range(0, len(sol.set_vector)):
            if random.random() < mut_prob:
                new_sol[i] = 1 if new_sol[i] == 0 else 0
        return Solution(self.problem_instance, new_sol)

    def iteration(self):
        mutated_pop = list(map(lambda x: self.mutate(x), self.population))
        mutated_pop = list(filter(lambda x: not self.dominated_by_any(self.population, x), mutated_pop))
        self.population = list(filter(lambda x: not self.dominated_by_any(mutated_pop, x), self.population))
        self.population.extend(mutated_pop)
        self.iter = self.iter + 1

    def set_logging(self, logger):
        self.logger = logger

    def find_approximation(self):
        print("start base GCAIS")
        old_best = sys.maxsize
        found = 0
        s = time.time()
        for i in range(0, self.iterations):
            if i % 10 == 0:
                print_now()
                print("finished " + str(i / self.iterations) + " percent of base GCAIS")
            iter_start = time.time()
            self.iteration()
            iter_end = time.time()
            feasible = list(filter(lambda x: x.is_feasible, self.population))
            try:
                best = min(feasible, key=lambda x: x.cost).cost
            except Exception as e:
                best = sys.maxsize
            if best != sys.maxsize:
                if best < old_best:
                    found = self.iter
                    old_best = best
                else:
                    if has_converged(found, self.iter, time.time() - s):
                        break
            self.logger.log_entry(self.iter, best, float(iter_end - iter_start), len(self.population))
        feasible = list(filter(lambda x: x.is_feasible, self.population))
        if len(feasible) == 0:
            all_sets = np.ones(self.problem_instance.problem_instance.shape[0])
            return Solution(self.problem_instance, all_sets)
        print("finished base GCAIS")
        return min(feasible, key=lambda x: x.cost)


class GCAIS:

    BORDER = 200
    # to tackle exploding population
    CROPPING = True

    def __init__(self, problem_instance, iterations):
        self.name = "GCAIS"
        self.population = {}
        self.problem_instance = problem_instance
        self.iterations = iterations
        number_of_sets = self.problem_instance.problem_instance.shape[0]
        sol_vector = np.ones(number_of_sets)
        self.population[0] = {}
        self.population[0]["sols"] = [Solution(self.problem_instance, sol_vector)]
        self.population[0]["best_covered"] = 0
        self.iter = 0

    def mutate(self, sol):
        mut_prob = 1.0 / len(sol.set_vector)
        new_sol = copy.deepcopy(sol.set_vector)
        for i in range(0, len(sol.set_vector)):
            if random.random() < mut_prob:
                new_sol[i] = 1 if new_sol[i] == 0 else 0
        return Solution(self.problem_instance, new_sol)

    def iteration(self):
        keys = list(map(lambda x: int(x), self.population.keys()))
        # print(len(keys))
        for key in keys:
            # print("started key " + str(key) + " with " + str(len(self.population[key]["sols"])) + " elements")
            to_add = list(map(lambda x: self.mutate(x), self.population[key]["sols"]))
            for ele in to_add:
                try:
                    if self.population[ele.cost]["best_covered"] < ele.covered:
                        self.population[ele.cost]["best_covered"] = ele.covered
                        self.population[ele.cost]["sols"] = [ele]
                    elif self.population[ele.cost]["best_covered"] == ele.covered:
                        if len(self.population[ele.cost]["sols"]) < GCAIS.BORDER or not GCAIS.CROPPING:
                            equals = False
                            for other_ele in self.population[ele.cost]["sols"]:
                                if ele.equals_other_sol(other_ele):
                                    equals = True
                                    break
                            if not equals:
                                self.population[ele.cost]["sols"].append(ele)
                except Exception as _:
                        self.population[ele.cost] = {}
                        self.population[ele.cost]["best_covered"] = ele.covered
                        self.population[ele.cost]["sols"] = [ele]
        keys = list(map(lambda x: int(x), self.population.keys()))
        for key in keys:
            try:
                for i in range(key + 1, max(keys) + 1):
                    if self.population[i]["best_covered"] <= self.population[key]["best_covered"]:
                        del self.population[i]
            except Exception as _:
                pass
        self.iter = self.iter + 1

    def set_logging(self, logger):
        self.logger = logger

    def find_approximation(self):
        print("start base GCAIS")
        old_best = sys.maxsize
        found = 0
        s = time.time()
        for i in range(0, self.iterations):
            if i % 10 == 0:
                print_now()
                print("finished " + str(i / self.iterations) + " percent of base GCAIS")
            iter_start = time.time()
            self.iteration()
            iter_end = time.time()
            feasible = []
            for key in self.population.keys():
                feasible.extend(list(filter(lambda x: x.is_feasible, self.population[key]["sols"])))
            try:
                best = min(feasible, key=lambda x: x.cost).cost
            except Exception as e:
                best = sys.maxsize
            if best != sys.maxsize:
                if best < old_best:
                    found = self.iter
                    old_best = best
                else:
                    if has_converged(found, self.iter, time.time() - s):
                        break
            self.logger.log_entry(self.iter, best, float(iter_end - iter_start), len(self.population))
        feasible = []
        for key in self.population.keys():
            feasible.extend(list(filter(lambda x: x.is_feasible, self.population[key]["sols"])))
        if len(feasible) == 0:
            all_sets = np.ones(self.problem_instance.problem_instance.shape[0])
            return Solution(self.problem_instance, all_sets)
        print("finished base GCAIS")
        return min(feasible, key=lambda x: x.cost)


class TestGCAIS(unittest.TestCase):

    def test_adapt_and_solve(self):

        class Logger:
            def __init__(self):
                pass

            def log_entry(self, iteration, best, time, popsize=None):
                pass

        a = np.zeros((5, 6))
        a[0] = [1, 1, 0, 0, 0, 0]
        a[1] = [0, 1, 1, 0, 0, 0]
        a[2] = [0, 0, 1, 1, 0, 0]
        a[3] = [0, 0, 0, 1, 1, 0]
        a[4] = [0, 0, 0, 0, 1, 1]
        sc = SetCover(a)
        ais = BoundedGCAIS(sc, 10, 20)
        ais.logger = Logger()
        ais.find_approximation()
        for _ in range(0, 20):
            sc, _, _, del_sets, _ = sc.mutate(3, 3, 3, 3, 0.5)
            ais.adapt_to_new_problem(sc, del_sets)
            ais.find_approximation()


if "__main__" == __name__:
    unittest.main()
