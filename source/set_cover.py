import numpy as np
import sys
import random
import unittest
from datetime import datetime
try:
    from source.converter import *
except:
    from converter import *

def has_converged(found, current, duration, adapted=False):
    factor = 1
    if adapted:
        factor = 2
    if found != -1:
        if current - found > (100 / factor):
            return True
    if duration > (300 / factor):
        return True
    return False

def print_now():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)


class SetCover:

    RAND_COSTS = None

    '''
    Represents an instance of the Minimum Set Cover problem
    '''
    def __init__(self, problem_instance, cost_func=None, create_rand=False):
        '''
        :param problem_instance: two dimensional numpy array. The rows represent the available
        sets and the columns the possible elements. If problem_instance[i][j] is one then i-th
        set has the j-th element. If it is set to 0 then the set does not have this element.
        '''
        self.problem_instance = problem_instance
        self.cost_func = None
        self.is_solveable()
        if create_rand:
            if SetCover.RAND_COSTS is None:
                SetCover.RAND_COSTS = load_and_clean("runtimes" + os.sep + "TestHist_51.json")
            self.cost_func = np.ones(self.problem_instance.shape[0])
            for i in range(0, len(self.cost_func)):
                weight_index = random.randint(0, len(SetCover.RAND_COSTS) - 1)
                self.cost_func[i] = SetCover.RAND_COSTS[weight_index]
        elif create_rand is False and cost_func is None:
            self.cost_func = None
        else:
            self.cost_func = cost_func

    def get_cost(self, index):
        if self.cost_func is None:
            # unicost
            return 1
        else:
            return self.cost_func[index]

    def is_solveable(self):
        '''
        Checks if there exists a set cover at all. Raises an Exception if not.
        '''
        all_sets = np.ones(self.problem_instance.shape[0])
        solution = Solution(self, all_sets)
        if not solution.is_feasible_solution():
            raise Exception("Set cover instance cannot be solved")

    def __is_now_empty_set(self, row_index, dropped_elements):
        row = self.problem_instance[row_index]
        for i in range(0, len(row)):
            if i not in dropped_elements:
                if row[i] == 1:
                    return False
        return True

    def __is_now_uncoverable(self, column, dropped_sets):
        for i in range(0, self.problem_instance.shape[0]):
            if i not in dropped_sets:
                if self.problem_instance[i][column] == 1:
                    return False
        return True

    def adapt_to_deletion(self, dropped_sets, dropped_elements):
        old_shape = self.problem_instance.shape
        empty_sets = list(filter(lambda x: self.__is_now_empty_set(x, dropped_elements), list(range(0, old_shape[0]))))
        dropped_sets.extend(empty_sets)
        dropped_sets = list(set(dropped_sets))
        uncoverable = list(filter(lambda x: self.__is_now_uncoverable(x, dropped_sets), list(range(0, old_shape[1]))))
        dropped_elements.extend(uncoverable)
        dropped_elements = list(set(dropped_elements))
        new_amount_sets = old_shape[0] - len(dropped_sets)
        new_amount_ele = old_shape[1] - len(dropped_elements)
        if new_amount_sets == 0 or new_amount_ele == 0:
            return None
        new_instance = np.zeros((new_amount_sets, new_amount_ele))
        new_cost_func = None
        if self.cost_func is not None:
            new_cost_func = np.zeros(new_amount_sets)
        row_counter = 0
        for i in range(0, old_shape[0]):
            column_counter = 0
            if i not in dropped_sets:
                for j in range(0, old_shape[1]):
                    if j not in dropped_elements:
                        entry = self.problem_instance[i][j]
                        new_instance[row_counter][column_counter] = entry
                        column_counter += 1
                if new_cost_func is not None:
                    new_cost_func[row_counter] = self.cost_func[row_counter]
                row_counter += 1
        if new_cost_func is not None:
            for i in range(row_counter, len(new_cost_func)):
                weight_index = random.randint(0, len(SetCover.RAND_COSTS) - 1)
                new_cost_func[i] = SetCover.RAND_COSTS[weight_index]
        return new_instance, new_cost_func

    @staticmethod
    def adapt_to_insertion(cleaned_mat, added_sets, added_elements, cost_func):
        old_shape = cleaned_mat.shape
        new_instance = np.zeros((old_shape[0] + len(added_sets), old_shape[1] + added_elements))
        if cost_func is not None:
            new_cost_func = np.zeros(old_shape[0] + len(added_sets))
        else:
            new_cost_func = None
        for i in range(0, old_shape[0]):
            for j in range(0, old_shape[1]):
                new_instance[i][j] = cleaned_mat[i][j]
            if cost_func is not None:
                new_cost_func[i] = cost_func[i]
        for i in range(0, len(added_sets)):
            for j in range(0, new_instance.shape[1]):
                new_instance[i + old_shape[0]][j] = added_sets[i][j]
            if cost_func is not None:
                weight_index = random.randint(0, len(SetCover.RAND_COSTS) - 1)
                new_cost_func[i + old_shape[0]] = SetCover.RAND_COSTS[weight_index]
        return new_instance, new_cost_func

    def adapt_problem_mat(self, dropped_sets, dropped_elements, added_sets, added_elements):
        new_ins, new_cost_func = self.adapt_to_deletion(dropped_sets, dropped_elements)
        new_ins, new_cost_func = SetCover.adapt_to_insertion(new_ins, added_sets, added_elements, new_cost_func)
        self.problem_instance = new_ins
        self.cost_func = new_cost_func
        self.is_solveable()

    @staticmethod
    def create_drops(amount, max_val):
        to_drop = []
        while len(set(to_drop)) != amount:
            to_drop.append(random.randint(0, max_val - 1))
        return list(set(to_drop))

    def mutate(self, max_del_sets, max_del_ele, max_add_sets, max_add_ele, mut_prob):
        feasible = False
        new_instance = None
        new_cost_func = None
        while not feasible:
            n_del_sets = random.randint(0, max_del_sets)
            n_del_ele = random.randint(0, max_del_ele)
            del_sets = SetCover.create_drops(n_del_sets, self.problem_instance.shape[0])
            del_ele = SetCover.create_drops(n_del_ele, self.problem_instance.shape[1])
            new_instance, new_cost_func = self.adapt_to_deletion(del_sets, del_ele)
            try:
                set_cover = SetCover(new_instance, new_cost_func)
                feasible = True
            except Exception:
                pass
        try:
            sets_to_add = random.randint(1, max_add_sets)
        except Exception:
            sets_to_add = 1
        ele_to_add = random.randint(0, max_add_ele)
        new_sets = np.zeros((sets_to_add, ele_to_add + new_instance.shape[1]))
        # set single requirements
        for i in range(0, new_sets.shape[0]):
            for j in range(0, new_sets.shape[1]):
                if random.random() < mut_prob:
                    new_sets[i][j] = 1
        for j in range(0, ele_to_add):
            if sets_to_add == 0:
                break
            covered = 0
            for i in range(0, sets_to_add):
                if new_sets[i][new_instance.shape[1] + j] == 1:
                    covered += 1
            if covered == 0:
                index = random.randint(0, sets_to_add - 1)
                new_sets[index][new_instance.shape[1] + j] = 1
        del_sets = list(set(del_sets))
        del_ele = list(set(del_ele))
        new_prob_mat, new_cost_func = SetCover.adapt_to_insertion(new_instance, new_sets, ele_to_add, new_cost_func)
        return SetCover(new_prob_mat, new_cost_func), sets_to_add, ele_to_add, del_sets, del_ele


class Solution:
    '''
    Represents a possible infeasible solution of the Set Cover problem
    '''
    def __init__(self, set_cover_instance, set_vector, is_feasible=None, cost=None):
        '''
        :param set_cover_instance: instance of SetCover
        :param set_vector: numpy vector indicating the sets that the solution holds.
        The i-th entry of set_vector corresponds to the i-th row of the set cover
        table.
        :param is_feasible: indicates if the solution is a possible set cover.
        :param cost: number of sets in the cover.
        '''
        self.set_cover_instance = set_cover_instance
        self.set_vector = set_vector
        self.is_feasible = is_feasible
        self.cost = cost
        self.is_feasible_solution()

    def adapt_to_mutation(self, new_instance, deleted_sets):
        self.set_cover_instance = new_instance
        self.is_feasible = None
        self.cost = None
        new_set_vector = np.zeros(new_instance.problem_instance.shape[0])
        count = 0
        for i in range(0, len(self.set_vector)):
            if i not in deleted_sets:
                new_set_vector[count] = self.set_vector[i]
                count += 1
        self.set_vector = new_set_vector
        self.is_feasible_solution()

    def equals_other_sol(self, other_sol):
        for i in range(0, len(self.set_vector)):
            if self.set_vector[i] != other_sol.set_vector[i]:
                return False
        return True

    def add_set(self, index):
        '''
        Adds the set of the given index to the solution. Afterwards the cost is updated
        and is checked if the solution becomes feasible.

        :param index: index in the set cover table of the set.
        '''
        if self.set_vector[index] == 1:
            return False
        self.set_vector[index] = 1
        self.cost += self.set_cover_instance.get_cost(index)
        self.covered_elements += self.set_cover_instance.problem_instance[index]
        self.covered_elements = [1 if ele > 0 else 0 for ele in self.covered_elements]
        if sum(self.covered_elements) == self.set_cover_instance.problem_instance.shape[1]:
            self.is_feasible = True
        self.covered = sum(self.covered_elements)
        return True

    def get_cost(self):
        if self.is_feasible():
            return self.cost
        else:
            return sys.maxsize

    def is_feasible_solution(self):
        '''
        Also retrieves the covered elements and calculates the solutions cost.
        '''
        if self.is_feasible is not None:
            return self.is_feasible
        available_elements = np.zeros(len(self.set_cover_instance.problem_instance[0]))
        cost = 0
        for i in range(0, len(self.set_vector)):
            if self.set_vector[i] == 1:
                cost += self.set_cover_instance.get_cost(i)
                available_elements += self.set_cover_instance.problem_instance[i]
        self.covered_elements = [1 if ele > 0 else 0 for ele in available_elements]
        self.covered = sum(self.covered_elements)
        if len(available_elements[0 in available_elements]) == 0:
            self.cost = cost
            self.is_feasible = True
            return self.is_feasible
        self.cost = cost
        self.is_feasible = False
        return self.is_feasible


class TestSetCover(unittest.TestCase):

    def test_mutation_problem_instance(self):
        a = np.zeros((5, 6))
        a[0] = [1, 1, 0, 0, 0, 0]
        a[1] = [0, 1, 1, 0, 0, 0]
        a[2] = [0, 0, 1, 1, 0, 0]
        a[3] = [0, 0, 0, 1, 1, 0]
        a[4] = [0, 0, 0, 0, 1, 1]
        sc = SetCover(a)
        for _ in range(0, 100):
            other, sets_to_add, ele_to_add, del_sets, del_ele = sc.mutate(3, 3, 3, 3, 0.5)
            new_matrix = other.problem_instance
            row_count = 0
            for i in range(0, 5):
                if i not in del_sets:
                    column_count = 0
                    for j in range(0, 6):
                        if j not in del_ele:
                            assert a[i][j] == new_matrix[row_count][column_count]
                            column_count += 1
                    row_count += 1

    def test_mutation_problem_sol_vector(self):
        a = np.zeros((5, 6))
        a[0] = [1, 1, 0, 0, 0, 0]
        a[1] = [0, 1, 1, 0, 0, 0]
        a[2] = [0, 0, 1, 1, 0, 0]
        a[3] = [0, 0, 0, 1, 1, 0]
        a[4] = [0, 0, 0, 0, 1, 1]
        sc = SetCover(a)
        for _ in range(0, 100):
            sol_vector = np.zeros(5)
            sol_vector[0] = 1
            sol_vector[2] = 1
            sol_vector[4] = 1
            solution = Solution(sc, sol_vector)
            other, _, _, del_sets, del_ele = sc.mutate(3, 3, 3, 3, 0.5)
            solution.adapt_to_mutation(other, del_sets)
            new_cost = 3 - len(list(filter(lambda x: x in [0, 2, 4], del_sets)))
            for i in [0, 2, 4]:
                if i not in del_sets:
                    new_index = i - len(list(filter(lambda x: x < i, del_sets)))
                    assert solution.set_vector[new_index] == 1
            assert solution.cost == new_cost

if "__main__" == __name__:
    pass
    # unittest.main()
