import time
import brute_force_search as bfs
import numpy as np


def benchmark_case(amount):
    data = np.load('Tests/case_{}.npy'.format(amount))
    query = np.load('Tests/query_{}.npy'.format(amount))
    print('{} points test'.format(amount))

    stopwatch = 0

    for index in range(0, 10, 1):
        stopwatch += benchmark(data, query[index], index+1, amount)

    print('{} points case ends in {} seconds'.format(amount, stopwatch))
    np.save('Output/Brute_force/Brute_force_Time_{}.npy'.format(amount), np.array(stopwatch/3))
    return stopwatch


#@profile
def benchmark(data, query, attempt, amount):
    start = time.perf_counter()
    output = bfs.knn_search(data, query, 100)[0]
    end = time.perf_counter()

    # print(np.sort(output))
    time_result = end - start
    np.save('Output/Brute_force/Brute_force_{}_{}.npy'.format(amount, attempt), output)

    del output, start, end
    print('attempt {} ends in {} seconds'.format(attempt, time_result))
    return time_result


def run_benchmark():
    print('Running "Bruteforce search" benchmark')
    stopwatch = 0
    # stopwatch += benchmark_case(1000)
    #stopwatch += benchmark_case(1000)
    #stopwatch += benchmark_case(10000)
    stopwatch += benchmark_case(100000)
    print('"Bruteforce search" benchmark ends in {} seconds'.format(stopwatch))
    print('===============================================================')