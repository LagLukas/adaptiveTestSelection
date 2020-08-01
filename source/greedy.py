try:
    from source.set_cover import SetCover
    from source.set_cover import Solution
    from source.gcais_population import GCAISPopulation
except Exception as _:
    from set_cover import SetCover
    from set_cover import Solution
    from gcais_population import GCAISPopulation
from abc import abstractmethod
import numpy as np
import copy
import time
import os


class FeasibleSolutionConstructor:

    @abstractmethod
    def make_solution_feasible(self, solution):
        pass


class GreedyAlgorithm(FeasibleSolutionConstructor):
    '''
    Greedy algorithm that always takes the set next which covers the most, extant elements.
    '''

    def __init__(self, set_cover_instance):
        self.name = "GREEDY"
        self.set_cover_instance = set_cover_instance
        self.population = []

    def get_best_next_set(self, table, already_covered, solution):
        '''
        retrieves the index of set that has the most uncovered elements.

        :param table: set cover instance table (rows = sets, columns = elements).
        :param already_covered: vector containing the already covered elements. Indexes
        correspond to the column indexes.

        :return : index
        '''
        max_index = -1
        max_val = -1
        for i in reversed(range(0, table.shape[0])):
            if solution.set_vector[i] != 1:
                return i
            '''
            element_count = sum([1 if table[i][j] > already_covered[j] else 0 for j in range(0, len(already_covered))])
            if element_count > max_val:
                max_val = element_count
                max_index = i
            '''
        return max_index

    def greedy_iteration(self, solution, table):
        biggest_set_index = self.get_best_next_set(table, solution.covered_elements, solution)
        solution.add_set(biggest_set_index)
        self.population.append(copy.deepcopy(solution))
        return solution.is_feasible

    def make_solution_feasible(self, solution):
        '''
        Applies the greedy algorithm on a infeasible solution to make feasible.
        '''
        while not solution.is_feasible:
            self.greedy_iteration(solution, self.set_cover_instance.problem_instance)

    def get_greedy_solution(self):
        empty_vec = np.zeros(self.set_cover_instance.problem_instance.shape[0])
        approx_sol = Solution(self.set_cover_instance, empty_vec)
        self.make_solution_feasible(approx_sol)
        return approx_sol

    def save_pop_results(self):
        pop = self.population
        GCAISPopulation.convert_and_save("PARETO_FRONTIERS" + os.sep + self.logger.get_file_name(), self.set_cover_instance, pop)

    def set_logging(self, logger):
        self.logger = logger

    def find_approximation(self):
        print("start greedy selection")
        iter_start = time.time()
        full_coverage = self.get_greedy_solution()
        iter_end = time.time()
        self.logger.log_entry(0, full_coverage.cost, float(iter_end - iter_start), len(self.population))
        self.save_pop_results()
        print("finish greedy selection")
        return full_coverage
