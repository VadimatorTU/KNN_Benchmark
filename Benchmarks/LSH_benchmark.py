import time
import lsh
import numpy as np


def benchmark_case(amount, nb_projections=10, nb_tables=10, quantization=10):
    data = np.load('Tests/case_{}.npy'.format(amount))
    query = np.load('Tests/query_{}.npy'.format(amount))
    print('{} points test'.format(amount))

    start = time.perf_counter()
    built_hash = lsh.LSH(nb_projections, nb_tables, quantization)
    built_hash.build(data)
    end = time.perf_counter()
    stopwatch = end - start
    print('built in {} seconds'.format(stopwatch))

    stopwatch += benchmark(built_hash, query[0], 1)
    stopwatch += benchmark(built_hash, query[1], 2)
    stopwatch += benchmark(built_hash, query[2], 3)

    print('{} points case ends in {} seconds'.format(amount, stopwatch))
    return stopwatch


# @profile
def benchmark(built_hash, query, attempt):
    start = time.perf_counter()
    output = built_hash.search(query, 100)[1]
    end = time.perf_counter()

    print(np.sort(output))
    time_result = end - start

    del start, end, output
    print('attempt {} ends in {} seconds'.format(attempt, time_result))
    return time_result


def run_benchmark():
    print('Running "LSH" benchmark')
    stopwatch = 0
    # stopwatch += benchmark_case(100)
    stopwatch += benchmark_case(1000)
    stopwatch += benchmark_case(10000)
    stopwatch += benchmark_case(100000)
    print('"LSH" benchmark ends in {} seconds'.format(stopwatch))
    print('===============================================================')
