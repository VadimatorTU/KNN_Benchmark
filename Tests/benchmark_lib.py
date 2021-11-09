import numpy as np


def generate_case(point, dim):
    case = np.random.rand(point, dim)
    np.save('case_{}.npy'.format(point), case)
    del case
    query = np.random.rand(3, dim)
    np.save('query_{}.npy'.format(point), query)
    del query


generate_case(100, 10)
generate_case(1000, 10)
generate_case(10000, 10)
generate_case(100000, 10)

