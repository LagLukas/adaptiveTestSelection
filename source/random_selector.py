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
import unittest


class RandomSelector:

    RETRIES = 5

    def __init__(self, problem_instance, time_budgets):
        self.name = "random"
        self.problem_instance = problem_instance
        self.time_budgets = time_budgets
        number_of_sets = problem_instance.problem_instance.shape[0]
        # get total execution time
        sol_vector = np.ones(number_of_sets)
        initial_sol = Solution(problem_instance, sol_vector)
        cost = initial_sol.cost
        self.time_budgets = list(map(lambda x: x * cost, self.time_budgets))
        self.population = []

    def draw_random_sol(self):
        index = random.randint(0, self.problem_instance.problem_instance.shape[0] - 1)
        cost = self.problem_instance.get_cost(index)
        return index, cost

    def get_sol_of_cost(self, budget):
        total = 0
        already = []
        re = 0
        while re < RandomSelector.RETRIES:
            index, set_cost = self.draw_random_sol()
            if index not in already and set_cost + total < budget:
                already.append(index)
                total += set_cost
                re = 0
            else:
                re += 1
        number_of_sets = self.problem_instance.problem_instance.shape[0]
        sol_vector = np.zeros(number_of_sets)
        initial_sol = Solution(self.problem_instance, sol_vector)
        for index in already:
            # initial_sol.set_vector[index] = 1
            initial_sol.add_set(index)
        return initial_sol

    def get_random_cover(self):
        number_of_sets = self.problem_instance.problem_instance.shape[0]
        sol_vector = np.zeros(number_of_sets)
        initial_sol = Solution(self.problem_instance, sol_vector)
        while not initial_sol.is_feasible_solution():
            index = random.randint(0, self.problem_instance.problem_instance.shape[0] - 1)
            initial_sol.add_set(index)
        return initial_sol

    def save_pop_results(self):
        pop = self.population
        GCAISPopulation.convert_and_save("PARETO_FRONTIERS" + os.sep + self.logger.get_file_name(), self.problem_instance, pop)

    def set_logging(self, logger):
        self.logger = logger

    def find_approximation(self):
        print("start random selection")
        iter_start = time.time()
        for budget in self.time_budgets:
            self.population.append(self.get_sol_of_cost(budget))
        full_coverage = self.get_random_cover()
        self.population.append(full_coverage)
        iter_end = time.time()
        self.logger.log_entry(0, full_coverage.cost, float(iter_end - iter_start), len(self.population))
        self.save_pop_results()
        print("finish random selection")
        return full_coverage
