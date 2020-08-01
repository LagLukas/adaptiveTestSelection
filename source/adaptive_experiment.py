import multiprocessing
import json
import os
try:
    from source.excel_reader import *
    from source.set_cover import *
    from source.mylogging import *
    from source.GCAIS import *
    from source.GSEMO import *
    from source.beasley_reader import BeasleyReader
    from source.random_selector import RandomSelector
    from source.seip import SEIP
except Exception as _:
    from excel_reader import *
    from set_cover import *
    from mylogging import *
    from GCAIS import *
    from GSEMO import *
    from beasley_reader import *
    from random_selector import RandomSelector
    from seip import SEIP


class Evaluation:

    PROCESSES = 10
    ITERATIONS = 40
    REPETITIONS = 100
    BSH = ["fridge1.xlsx", "fridge2.xlsx", "fridge3.xlsx"]
    BEASLEY_DATASETS = []

    def __init__(self, myseed=3141):
        random.seed(myseed)
        self.experiments = []
        mut_dict = {}
        mut_dict["set_del"] = 0.025
        mut_dict["del_ele"] = 0.025
        mut_dict["set_add"] = 0.05
        mut_dict["ele_add"] = 0.05
        mut_dict["mut_prob"] = 0.05
        # load bsh shite
        for pol in Evaluation.BSH:
            sc = SetCover(PolarionReader("DATA" + os.sep + pol).create_np_matrix(), None, True)
            name = pol[:-5] + ".json"
            problem_name = "DATA" + os.sep + name
            self.experiments.append(AdaptiveExperiment(sc, problem_name, Evaluation.ITERATIONS, mut_dict))
        # load beasley
        for beasley in Evaluation.BEASLEY_DATASETS:
            prob_path = "DATA" + os.sep + beasley
            sc = BeasleyReader(prob_path).read_file()
            name = beasley[:-4] + ".json"
            problem_name = "DATA" + os.sep + name
            self.experiments.append(AdaptiveExperiment(sc, problem_name, Evaluation.ITERATIONS, mut_dict))

    def perform_repetition(self, rep):
        for experiment in self.experiments:
            experiment.run_exp_rep(rep)

    def start_experiments(self, parallel):
        if parallel:
            p = multiprocessing.Pool(Evaluation.PROCESSES)
            avg_res = p.map(self.perform_repetition, range(Evaluation.REPETITIONS))
        else:
            for i in range(0, Evaluation.REPETITIONS):
                self.perform_repetition(i)


class AdaptiveExperiment:

    TIME_BUDGETS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    ALGORITHMS = [
        lambda sci: BoundedGCAIS(sci, 5000, 200, True, True, True),
        lambda sci: RandomSelector(sci, AdaptiveExperiment.TIME_BUDGETS),
        lambda sci: BoundedGCAIS(sci, 5000, 200, True, True),
        lambda sci: BoundedGCAIS(sci, 5000, 200, False, False),
        lambda sci: Simple_GSEMO(sci, 30, 200000),
        lambda sci: Simple_GSEMO(sci, 1, 200000),
        lambda sci: SEIP(sci, 2000)
    ]

    def __init__(self, problem_instance, problem_name, iterations, mut_dict):
        self.original_problem_instance = problem_instance
        self.problem_name = problem_name
        self.iterations = iterations
        self.mut_dict = mut_dict
        self.create_mutated_instances()

    def create_mutated_instances(self):
        self.mutated_instances = []
        scp = self.original_problem_instance
        for _ in range(0, self.iterations):
            m = scp.problem_instance.shape[0]  # sets
            n = scp.problem_instance.shape[1]  # elements
            max_del_sets = int(m * self.mut_dict["set_del"])
            max_del_ele = int(n * self.mut_dict["del_ele"])
            max_add_sets = int(m * self.mut_dict["set_add"])
            max_add_ele = int(n * self.mut_dict["ele_add"])
            mut_prob = self.mut_dict["mut_prob"]
            scp, sets_to_add, ele_to_add, del_sets, del_ele = scp.mutate(max_del_sets, max_del_ele, max_add_sets, max_add_ele, mut_prob)
            adapted_dict = {}
            adapted_dict["instance"] = scp
            adapted_dict["sets_to_add"] = sets_to_add
            adapted_dict["ele_to_add"] = ele_to_add
            adapted_dict["del_sets"] = del_sets
            adapted_dict["del_ele"] = del_ele
            self.mutated_instances.append(adapted_dict)

    def get_instance(self, index):
        if index == 0:
            return self.original_problem_instance
        elif index < self.iterations:
            return self.mutated_instances[index - 1]["instance"]

    def get_adaptation_parameters(self, index):
        if index > 0 and index < self.iterations:
            return self.mutated_instances[index - 1]

    def get_logger(self, algo_name, adaptation_index, repetition_index):
        '''
        :param algo_name: name of the method
        :param adaptation_index: index of the problem instance
        :param repetition_index: repetition index
        '''
        name = "FINE_RESULTS" + os.sep + self.problem_name.split(os.sep)[-1][:-5] + "_" + \
               algo_name + "_" + str(adaptation_index) + "_" + str(repetition_index) + ".json"
        return Logging(name)

    def run_exp_algo(self, algo_index, repetition):
        algo = AdaptiveExperiment.ALGORITHMS[algo_index](self.original_problem_instance)
        algo.logger = self.get_logger(algo.name, 0, repetition)
        algo.find_approximation()
        algo.logger.save()
        for i in range(1, self.iterations):
            adaptation_info = self.get_adaptation_parameters(i)
            if isinstance(algo, BoundedGCAIS):
                if algo.adaptive:
                    algo.adapt_to_new_problem(adaptation_info["instance"], adaptation_info["del_sets"])
                else:
                    algo = AdaptiveExperiment.ALGORITHMS[algo_index](self.get_instance(i))
            else:
                algo = AdaptiveExperiment.ALGORITHMS[algo_index](self.get_instance(i))
            algo.logger = self.get_logger(algo.name, i, repetition)
            algo.find_approximation()
            algo.logger.save()

    def run_exp_rep(self, rep):
        for i in range(0, len(AdaptiveExperiment.ALGORITHMS)):
            self.run_exp_algo(i, rep)


if "__main__" == __name__:
    eval = Evaluation()
    eval.start_experiments(True)