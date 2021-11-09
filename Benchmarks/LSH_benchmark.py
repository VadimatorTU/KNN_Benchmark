import time
import lsh
import numpy as np


def benchmark_case(amount, nb_projections=10, nb_tables=10, quantization=1):
    data = np.load('Tests/case_{}.npy'.format(amount))
    query = np.load('Tests/query_{}.npy'.format(amount))
    print('{} points test, {} projections, {} tables'.format(amount, nb_projections, nb_tables))

    start = time.perf_counter()
    built_hash = lsh.LSH(nb_projections, nb_tables, quantization)
    built_hash.build(data)
    end = time.perf_counter()
    stopwatch = end - start
    print('built in {} seconds'.format(stopwatch))

    for index in range(0, 10, 1):
        stopwatch += benchmark(built_hash, query[index], index+1, amount, nb_projections, nb_tables, quantization)

    print('{} points case ends in {} seconds'.format(amount, stopwatch))
    np.save('Output/LSH/LSH_Time_{}_{}_{}_{}.npy'.format(amount, nb_projections, nb_tables, quantization), np.array(stopwatch/3))

    return stopwatch


#@profile
def benchmark(built_hash, query, attempt, amount, nb_projections, nb_tables, quantization):
    start = time.perf_counter()
    output = built_hash.search(query, 100)[1]
    end = time.perf_counter()

    # print(np.sort(output))
    time_result = end - start

    np.save('Output/LSH/LSH_{}_{}_{}_{}_{}.npy'.format(amount, nb_projections, nb_tables, quantization, attempt), output)
    del start, end, output
    print('attempt {} ends in {} seconds'.format(attempt, time_result))
    return time_result


def run_benchmark():
    print('Running "LSH" benchmark')
    stopwatch = 0
    # stopwatch += benchmark_case(100)
    # stopwatch += benchmark_case(1000)
    # stopwatch += benchmark_case(10000)
    # stopwatch += benchmark_case(100000)

    for np_tables in range(2, 5, 1):
        benchmark_case(100000, 2, np_tables)

    print('===============================================================')

    for np_tables in range(2, 5, 1):
        benchmark_case(100000, 5, np_tables)

    print('===============================================================')

    for np_tables in range(2, 5, 1):
        benchmark_case(100000, 10, np_tables)

    print('"LSH" benchmark ends in {} seconds'.format(stopwatch))
    print('===============================================================')
