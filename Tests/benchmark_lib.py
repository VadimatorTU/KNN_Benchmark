import numpy as np
DIM = 20


def generate_case(point, dim):
    case = np.random.rand(point, dim)
    np.save('case_{}.npy'.format(point), case)
    del case
    query = np.random.rand(10, dim)
    np.save('query_{}.npy'.format(point), query)
    del query


generate_case(100, DIM)
generate_case(1000, DIM)
generate_case(10000, DIM)
generate_case(100000, DIM)
generate_case(1000000, DIM)

