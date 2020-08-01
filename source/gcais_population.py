import sys
import random
from math import floor
try:
    from source.set_cover import *
except Exception as _:
    from set_cover import *


class GCAISPopulation:

    def __init__(self, initial_sol, to_be_covered, border, keep=True):
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