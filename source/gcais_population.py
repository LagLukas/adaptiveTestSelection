import sys
import random
from math import floor
try:
    from source.set_cover import *
except Exception as _:
    from set_cover import *


class RandInitializer:

    RETRIES = 5

    def __init__(self, problem_instance, time_budgets=None):
        self.name = "random"
        self.problem_instance = problem_instance
        if time_budgets is None:
            self.time_budgets = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        else:
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
        while re < RandInitializer.RETRIES:
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

    def get_random_covers(self):
        random_covers = []
        for budget in self.time_budgets:
            random_covers.append(self.get_sol_of_cost(budget))
        return random_covers


class GCAISPopulation:

    def __init__(self, initial_sol, to_be_covered, border, keep=True, with_rand=False, cover_instance=None):
        self.table = {}
        self.to_be_covered = to_be_covered
        initial = initial_sol.covered
        self.table[initial] = {}
        self.table[initial]["sols"] = [initial_sol]
        self.table[initial]["cost"] = initial_sol.cost
        self.max_cost = initial_sol.cost
        self.key_gen = self.get_key_generator()
        self.border = border
        self.keep = keep
        self.with_rand = with_rand
        if with_rand:
            rand_selector = RandInitializer(cover_instance)
            covers = rand_selector.get_random_covers()
            for cover in covers:
                self.try_insert(cover)

    def save_rough_info(self, path, cover_instance):
        to_be_saved = {}
        to_be_saved["table"] = {}
        for key in self.table.keys():
            to_be_saved["table"][str(key)] = self.table[key]["cost"]
        to_be_saved["total_reqs"] = cover_instance.problem_instance.shape[1]
        # get total execution time
        number_of_sets = cover_instance.problem_instance.shape[0]
        sol_vector = np.ones(number_of_sets)
        initial_sol = Solution(cover_instance, sol_vector)
        to_be_saved["total_cost"] = initial_sol.cost
        with open(path, "w") as file:
            json.dump(to_be_saved, file, sort_keys=True, indent=4)

    @staticmethod
    def convert_and_save(path, cover_instance, solutions):
        to_be_covered = cover_instance.problem_instance.shape[1]
        border = sys.maxsize
        look_up = GCAISPopulation(solutions[0], to_be_covered, border)
        for i in range(1, len(solutions)):
            look_up.try_insert(solutions[i])
        look_up.save_rough_info(path, cover_instance)

    def adapt_to_new_problem(self, new_prob_instance, deleted_sets, to_be_covered, joshi):
        self.max_cost = new_prob_instance.problem_instance.shape[0]
        if not joshi:
            self.to_be_covered = to_be_covered
            old_pop = []
            for entry in self.table.keys():
                for ele in self.table[entry]["sols"]:
                    ele.adapt_to_mutation(new_prob_instance, deleted_sets)
                    old_pop.append(ele)
                    # only one element...
                    break
            self.table = {}
            first = old_pop[0]
            self.table[first.covered] = {}
            self.table[first.covered]["sols"] = [first]
            self.table[first.covered]["cost"] = first.cost
            for i in range(1, len(old_pop)):
                self.try_insert(old_pop[i])
            number_of_sets = new_prob_instance.problem_instance.shape[0]
            sol_vector = np.ones(number_of_sets)
            initial_sol = Solution(new_prob_instance, sol_vector)
            self.try_insert(initial_sol)
            self.clean()
        else:
            self.to_be_covered = to_be_covered
            try:
                first = self.table[self.to_be_covered]["sols"][0]
                first.adapt_to_mutation(new_prob_instance, deleted_sets)
            except Exception:
                number_of_sets = new_prob_instance.problem_instance.shape[0]
                sol_vector = np.ones(number_of_sets)
                first = Solution(new_prob_instance, sol_vector)
            self.table = {}
            self.table[first.covered] = {}
            self.table[first.covered]["sols"] = [first]
            self.table[first.covered]["cost"] = first.cost
            number_of_sets = new_prob_instance.problem_instance.shape[0]
            sol_vector = np.ones(number_of_sets)
            initial_sol = Solution(new_prob_instance, sol_vector)
            assert initial_sol.is_feasible
            self.try_insert(initial_sol)
            self.clean()
            has_feasible = False
            for key in self.table.keys():
                for sol in self.table[key]["sols"]:
                    has_feasible = has_feasible or sol.is_feasible
            assert has_feasible is True
        if self.with_rand:
            rand_selector = RandInitializer(new_prob_instance)
            covers = rand_selector.get_random_covers()
            for cover in covers:
                self.try_insert(cover)

    def get_population_size(self):
        total = 0
        for key in self.table.keys():
            total += len(self.table[key]["sols"])
        return total

    def get_key_generator(self):
        for key in self.table.keys():
            yield int(key)

    def try_insert(self, sol):
        cost = sol.cost
        try:
            if self.table[sol.covered]["cost"] > cost:
                self.table[sol.covered]["sols"] = [sol]
                self.table[sol.covered]["cost"] = sol.cost
                return True
            elif self.table[sol.covered]["cost"] == cost:
                self.table[sol.covered]["sols"].append(sol)
                if len(self.table[sol.covered]["sols"]) > self.border:
                    to_rem = random.randint(0, len(self.table[sol.covered]["sols"]) - 1)
                    del self.table[sol.covered]["sols"][to_rem]
                return True
            return False
        except Exception as _:
            self.table[sol.covered] = {}
            self.table[sol.covered]["cost"] = sol.cost
            self.table[sol.covered]["sols"] = [sol]
            return True

    def non_dominated(self, sol):
        cost = sol.cost
        insert = False
        try:
            if self.table[sol.covered]["cost"] > cost:
                insert = True
            elif self.table[sol.covered]["cost"] == cost:
                insert = True
        except Exception as _:
            insert = True
        return insert

    def clean(self):
        '''
        cleans up the table after (not threadsafe should be run sequentially)
        '''
        keys = list(map(lambda x: int(x), self.table.keys()))
        last = max(keys)
        for i in reversed(range(0, max(keys))):
            try:
                if self.table[last]["cost"] <= self.table[i]["cost"]:
                    if not self.keep:
                        del self.table[i]
                    else:
                        del self.table[i]["sols"][1:]
                else:
                    last = i
            except Exception as _:
                pass
        self.key_gen = self.get_key_generator()

    def get_pop_chunk(self):
        return self.table[next(self.key_gen)]["sols"]

    def get_best(self):
        try:
            return self.table[self.to_be_covered]["sols"][0]
        except Exception as _:
            return self.max_cost

    def get_best_cost(self):
        try:
            return self.table[self.to_be_covered]["sols"][0].cost
        except Exception as _:
            return self.max_cost
