import time
import kdtree
import numpy as np

@profile
def benchmark_case(amount, leaf_size=1):
    data = np.load('Tests/case_{}.npy'.format(amount))
    query = np.load('Tests/query_{}.npy'.format(amount))
    print('{} points test'.format(amount))

    start = time.perf_counter()
    built_hash = kdtree.KDTree(leaf_size)
    built_hash.build(data)
    end = time.perf_counter()
    stopwatch = end - start
    print('built in {} seconds'.format(stopwatch))

    for index in range(0, 10, 1):
        stopwatch += benchmark(built_hash, query[index], index+1, amount, leaf_size)

    print('{} points case ends in {} seconds'.format(amount, stopwatch))
    np.save('Output/KDTree/KDTree_Time_{}_{}.npy'.format(amount, leaf_size), np.array(stopwatch/10))

    return stopwatch


#@profile
def benchmark(built_hash, query, attempt, amount, leaf_size=1):
    start = time.perf_counter()
    output = built_hash.search(query, 100)[1]
    end = time.perf_counter()

    #print(np.sort(output))
    time_result = end - start
    np.save('Output/KDTree/KDTree_{}_{}_{}.npy'.format(amount, leaf_size, attempt), output)

    del start, end, output
    print('attempt {} ends in {} seconds'.format(attempt, time_result))
    return time_result


def run_benchmark():
    print('Running "KDTree" benchmark')
    stopwatch = 0
    # stopwatch += benchmark_case(100)
    #stopwatch += benchmark_case(1000)
    #stopwatch += benchmark_case(10000)
    #stopwatch += benchmark_case(100000)

    """for leaf_size in range(1, 11, 3):
        benchmark_case(1000000, leaf_size)"""

    #benchmark_case(100000, 1)
    #benchmark_case(100000, 4)
    benchmark_case(100000, 7)
    #benchmark_case(100000, 10)

    print('"KDTree" benchmark ends in {} seconds'.format(stopwatch))
    print('===============================================================')
