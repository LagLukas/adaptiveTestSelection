import json
import time
import multiprocessing
import numpy as np
import sys
import os
try:
    from source.beasley_reader import BeasleyReader
    from source.mylogging import Logging
    from source.set_cover import SetCover
    from source.cro import CRO
    from source.genetic_algorithm import GeneticAlgorithm
    from source.simulated_annealing import SimulatedAnnealing
    from source.nbpso import NBPSO
    from source.seip import SEIP
    from source.jpso import JPSO
    from source.GCAIS import *
    from source.GSEMO import *
    from source.excel_reader import *
except Exception as _:
    from beasley_reader import BeasleyReader
    from mylogging import Logging
    from set_cover import SetCover
    from cro import CRO
    from genetic_algorithm import GeneticAlgorithm
    from simulated_annealing import SimulatedAnnealing
    from nbpso import NBPSO
    from seip import SEIP
    from jpso import JPSO
    from GCAIS import *
    from GSEMO import *
    from excel_reader import *

# GLOBAL
DATASETS = ["DATA" + os.sep + "chess.json", "DATA" + os.sep + "mushroom.json"]#, r"DATA\connect.json"]
PROCESSES = 4
ITERATIONS = 10

ALGO_ITER = 1500000

# SA parameters
SA_ITERATIONS = ALGO_ITER

# GA parameters
GA_POP_SIZE = 200
GA_K = 2
GA_MUT = 0.05
GA_ITER = ALGO_ITER

# NBPSO parameters
NBPSO_POP_SIZE = 30
NBPSO_ITERATIONS = ALGO_ITER

# CRO parameters
CRO_POP_SIZE = 10
CRO_MOLE_COLL = 0.1
CRO_BUFFER = 10000
CRO_ALPHA = 10000
CRO_BETA = 1000
CRO_ITERATIONS = ALGO_ITER

# SEIP parameters
SEIP_ITERATIONS = ALGO_ITER

# JPSO parameters
JPSO_POP_SIZE = 30
JPSO_ITERATIONS = ALGO_ITER

# GCAIS parameters
GCAIS_ITERATIONS = ALGO_ITER

# GSEMO
GSEMO_ITERATIONS = ALGO_ITER
LOCAL_POPULATIONS = 30

ALGORITHMS = [
              lambda sci: GCAIS_BASE(sci, 2000),
              lambda sci: Simple_GSEMO(sci, 30, 20000),
              #lambda sci: EpsilonGCAIS(sci, 2000, 5),
              #lambda sci: EpsilonGCAIS(sci, 2000, 10),
              #lambda sci: EpsilonGCAIS(sci, 2000, 15),
              lambda sci: EpsilonBoundedGCAIS(sci, 2000, 200, -1),
              #lambda sci: EpsilonBoundedGCAIS(sci, 2000, 200, 5),
              #lambda sci: EpsilonBoundedGCAIS(sci, 2000, 200, 10),
              #lambda sci: EpsilonBoundedGCAIS(sci, 2000, 200, 15)
              ]


'''
ALGORITHMS = [#lambda sci: NBPSO(NBPSO_POP_SIZE, sci, NBPSO_ITERATIONS),
              lambda sci: SimulatedAnnealing(sci, SA_ITERATIONS),
              lambda sci: SEIP(sci, SEIP_ITERATIONS),
              lambda sci: GCAIS(sci, GCAIS_ITERATIONS),
              lambda sci: GSEMO(sci, LOCAL_POPULATIONS, GCAIS_ITERATIONS),
              lambda sci: JPSO(JPSO_POP_SIZE, sci, JPSO_ITERATIONS),
              lambda sci: GeneticAlgorithm(sci, GA_POP_SIZE, GA_K, GA_MUT, GA_ITER),
              lambda sci: CRO(sci, CRO_POP_SIZE, CRO_MOLE_COLL, CRO_BUFFER, CRO_ALPHA, CRO_BETA, CRO_ITERATIONS)
              ]
'''

def prepare_data(path):
    file = open(path)
    data = json.load(file)
    problem_matrix = np.array(data)
    return SetCover(problem_matrix)


PROBLEM_INSTANCES = {}
# 0 Steiner Tiples, 2 bad case for greedy, 3 for Beasley's OR lib
MODE = 3
SIMULATED_DATASETS = 4
SIMULATED_BAD_CASES = 5
SIMULATED_MAX_K = 5
SIMULATED_N = 5


if MODE == 0:
    # Mode 1 for Steiner Triples
    PROBLEM_INSTANCES["DATA" + os.sep + "Steiner_27.json"] = SteinerTriple(True)
    PROBLEM_INSTANCES["DATA" + os.sep + "Steiner_45.json"] = SteinerTriple(False)
if MODE == 1:
    # Mode 0 for random instances
    for j in range(0, SIMULATED_DATASETS):
        rand_instance = TightSetCover(2, SIMULATED_MAX_K, SIMULATED_N)
        n_ele = str(sum(rand_instance.number_ele))
        n_sets = str(sum(rand_instance.number_sets))
        name = "DATA" + os.sep + "rand_" + n_sets + "_" + n_ele + "_" + str(j) + ".json"
        PROBLEM_INSTANCES[name] = rand_instance
if MODE == 3:
    filez = list(filter(lambda x: ".txt" in x, os.listdir("DATA")))
    filez = list(filter(lambda x: "scpe" in x, filez))
    for file in filez:
        prob_path = "DATA" + os.sep + file
        prob_instance = BeasleyReader(prob_path).read_file()
        name = file[:-4] + ".json"
        PROBLEM_INSTANCES["DATA" + os.sep + name] = prob_instance
if MODE == 4:
    polarions = ["polarion.xlsx"]
    for polarion in polarions:
        prob_path = "DATA" + os.sep + polarion
        prob_instance = SetCover(PolarionReader(prob_path).create_np_matrix())
        name = polarion[:-5] + ".json"
        PROBLEM_INSTANCES["DATA" + os.sep + name] = prob_instance


#def has_converged(found, current):
#    assert current - found > 0
#    if current - found > 100:
#        return True
#    return False


def iter_problem(problem_instance, iteration, name):
    results = {}
    runtime_results = {}
    for alg_creator in ALGORITHMS:
        algo = alg_creator(problem_instance)
        loc_name = "FINE_RESULTS" + os.sep + name.split(os.sep)[-1][:-5] + "_" + algo.name + "_" + str(iteration) + ".json"
        logger = Logging(loc_name)
        algo.set_logging(logger)
        start = time.time()
        approx = algo.find_approximation()
        try:
            results[algo.name] = approx.cost
        except Exception as _:
            results[algo.name] = approx
        end = time.time()
        duration = float(end - start)
        runtime_results[algo.name] = duration
        logger.save()
    name = "RESULTS" + os.sep + name.split(os.sep)[-1][:-5] + "_" + str(iteration) + ".json"
    with open(name, "w") as file:
        json.dump(results, file, sort_keys=True, indent=4)
    name = "RUNTIME_RESULTS" + os.sep + name.split(os.sep)[-1][:-5] + "_" + str(iteration) + ".json"
    with open(name, "w") as file:
        json.dump(runtime_results, file, sort_keys=True, indent=4)


def experiment_iter(iteration):
    global PROBLEM_INSTANCES
    for problem in PROBLEM_INSTANCES.keys():
        iter_problem(PROBLEM_INSTANCES[problem], iteration, problem)


def start_experiments(exp_fun, parallel):
    if parallel:
        p = multiprocessing.Pool(PROCESSES)
        avg_res = p.map(exp_fun, range(ITERATIONS))
        print("hey")
    else:
        for i in range(0, ITERATIONS):
            exp_fun(i)


def run_set_cover_experiment(parallel=True):
    start_experiments(experiment_iter, parallel)


if __name__ == '__main__':
    run_set_cover_experiment()
